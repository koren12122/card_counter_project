import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import yt_dlp

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
VIDEO_URLS = [
    'https://www.youtube.com/watch?v=c-Hx0_Hq860&pp=ygUSYmxhY2tqYWNrIHNlc3Npb24g',
    'https://www.youtube.com/watch?v=W6FO1fIP6eE&pp=ygUSYmxhY2tqYWNrIHNlc3Npb24g',
    'https://www.youtube.com/watch?v=mknYV_8aEhg&pp=ygUSYmxhY2tqYWNrIHNlc3Npb24g0gcJCcUKAYcqIYzv',
    'https://www.youtube.com/watch?v=ZzJ7KcQJ82I&pp=ygUSYmxhY2tqYWNrIHNlc3Npb24g',
    'https://www.youtube.com/watch?v=md-aaMwYhNc&pp=ygUSYmxhY2tqYWNrIHNlc3Npb24g',
    'https://www.youtube.com/watch?v=V4r_OGiB-dg&pp=ygUSYmxhY2tqYWNrIHNlc3Npb24g',
    'https://www.youtube.com/watch?v=s9N6zz6NGvE&pp=ygUSYmxhY2tqYWNrIHNlc3Npb24g',
    'https://www.youtube.com/watch?v=9DRkL2yyvIE&pp=ygUSYmxhY2tqYWNrIHNlc3Npb24g',
    'https://www.youtube.com/watch?v=abg6nj02FFU&pp=ygUSYmxhY2tqYWNrIHNlc3Npb24g',
    'https://www.youtube.com/watch?v=MdRqrHSs7v8&pp=ygUSYmxhY2tqYWNrIHNlc3Npb24g',
    'https://www.youtube.com/watch?v=RiSQNXgKA28&pp=ygUSYmxhY2tqYWNrIHNlc3Npb24g',
    'https://www.youtube.com/watch?v=T0UVNZUbJKA&pp=ygUSYmxhY2tqYWNrIHNlc3Npb24g',
    'https://www.youtube.com/watch?v=hKFhqkXSjAk&t=11s&pp=ygUSYmxhY2tqYWNrIHNlc3Npb24g',
    'https://www.youtube.com/watch?v=HBwk0xpPF4M&pp=ygUSYmxhY2tqYWNrIHNlc3Npb24g',
    'https://www.youtube.com/watch?v=nLtFYsx-NX8&pp=ygUSYmxhY2tqYWNrIHNlc3Npb24g'
]

SNAPSHOT_INTERVAL = 5       # seconds between frame checks
DIFF_THRESHOLD = 100.0      # MSE threshold to consider frames "different enough"
MIN_SEC_DIFF = 5          # minimum seconds between saved frames
CONF_THRESHOLD = 0.5        # YOLO confidence to accept a card detection
IMG_SIZE = 640               # resize resolution (square)

SCRIPT_DIR = Path(__file__).parent
RAW_DIR = (SCRIPT_DIR / '..' / 'raw').resolve()
DETECTOR_PATH = (SCRIPT_DIR / '..' / 'weights' / 'detector.pt').resolve()

YDL_OPTS = {
    'format': 'best',
    'format_sort': ['res:720'],
    'merge_output_format': 'mp4',
    'quiet': True,
    'no_warnings': True,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def calculate_difference(image1, image2):
    """Return the Mean Squared Error between two BGR images."""
    if image1 is None or image2 is None:
        return float('inf')

    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    if gray1.shape != gray2.shape:
        gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))

    err = np.sum((gray1.astype("float") - gray2.astype("float")) ** 2)
    err /= float(gray1.shape[0] * gray1.shape[1])
    return err


def download_video(url: str, dest_dir: str) -> Path:
    """Download a YouTube video to *dest_dir* and return the file path."""
    opts = {
        **YDL_OPTS,
        'outtmpl': os.path.join(dest_dir, '%(id)s.%(ext)s'),
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
    return Path(filename)


def process_video(video_path: Path, output_dir: Path, model=None):
    """Extract frames from *video_path*, filter with MSE + YOLO, save to *output_dir*."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Error opening video file: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    check_frames = max(1, int(fps * SNAPSHOT_INTERVAL))
    video_name = video_path.stem
    out_dir = output_dir / video_name
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    saved_count = 0
    last_saved_frame = None
    last_saved_frame_count = -fps * MIN_SEC_DIFF

    print(f"  Extracting frames (FPS: {fps:.2f}, interval: {SNAPSHOT_INTERVAL}s)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % check_frames == 0:
            current_sec = frame_count / fps
            last_saved_sec = last_saved_frame_count / fps
            if (current_sec - last_saved_sec) < MIN_SEC_DIFF:
                frame_count += 1
                continue

            save_this_frame = False

            if DIFF_THRESHOLD > 0:
                diff = calculate_difference(last_saved_frame, frame)
                if diff > DIFF_THRESHOLD:
                    save_this_frame = True
            else:
                save_this_frame = True

            if save_this_frame and model is not None:
                resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                results = model(resized, imgsz=IMG_SIZE, verbose=False)
                obb_data = results[0].obb
                has_card = (
                    obb_data is not None
                    and len(obb_data)
                    and any(float(c) >= CONF_THRESHOLD for c in obb_data.conf)
                )
                if not has_card:
                    save_this_frame = False
                else:
                    frame = resized
            elif save_this_frame:
                frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

            if save_this_frame:
                log_suffix = f" (diff: {diff:.2f})" if DIFF_THRESHOLD > 0 else ""
                print(f"    Frame at {current_sec:.1f}s saved{log_suffix}")

                out_path = out_dir / f"frame_{saved_count:05d}_sec_{int(current_sec)}.jpg"
                cv2.imwrite(str(out_path), frame)
                last_saved_frame = frame.copy()
                last_saved_frame_count = frame_count
                saved_count += 1

        frame_count += 1

    cap.release()
    print(f"  Finished {video_path.name}. Saved {saved_count} frames.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    model = None
    if YOLO is None:
        print("Warning: ultralytics not installed — skipping YOLO card detection.")
    elif not DETECTOR_PATH.exists():
        print(f"Warning: detector not found at {DETECTOR_PATH} — skipping YOLO card detection.")
    else:
        print(f"Loading detector from {DETECTOR_PATH} ...")
        model = YOLO(str(DETECTOR_PATH))

    with tempfile.TemporaryDirectory() as tmp_dir:
        for idx, url in enumerate(VIDEO_URLS, start=1):
            print(f"[{idx}/{len(VIDEO_URLS)}] {url}")
            try:
                video_file = download_video(url, tmp_dir)
                print(f"  Downloaded -> {video_file.name}")
                process_video(video_file, RAW_DIR, model=model)
            except Exception as e:
                print(f"  Error: {e}")

    print(f"Done! Filtered snapshots saved to: {RAW_DIR}")


if __name__ == '__main__':
    main()
