"""
Card Tracking & Classification Pipeline (Apple Silicon Optimized)
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from collections import Counter, defaultdict

import yaml

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# ──────────────────────────────────────────────────────────────────────
# Threaded camera reader — eliminates I/O blocking from the main loop
# ──────────────────────────────────────────────────────────────────────
class CameraStream:
    """Continuously grabs frames on a background thread so the main
    loop never blocks waiting for the webcam."""

    def __init__(self, src: int = 0, backend: int | None = None,
                 width: int = 1280, height: int = 720):
        args = (src, backend) if backend is not None else (src,)
        self.cap = cv2.VideoCapture(*args)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._grabbed = False
        self._stopped = False

        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()

    def _update(self):
        while not self._stopped:
            grabbed, frame = self.cap.read()
            with self._lock:
                self._grabbed = grabbed
                self._frame = frame

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            return self._grabbed, self._frame.copy() if self._frame is not None else None

    def isOpened(self) -> bool:
        return self.cap.isOpened()

    def release(self):
        self._stopped = True
        self._thread.join(timeout=2.0)
        self.cap.release()

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# We use the raw .pt files — PyTorch MPS will handle the hardware acceleration
OBB_MODEL_PATH = os.path.join(SCRIPT_DIR, "weights", "detector.pt")
CLS_MODEL_PATH = os.path.join(SCRIPT_DIR, "weights", "med_cls.pt")

TRACKER_DIR = os.path.join(SCRIPT_DIR, "trackers")
TRACKER_CONFIGS = {
    "sort":      os.path.join(TRACKER_DIR, "sort.yaml"),
    "bytetrack": os.path.join(TRACKER_DIR, "bytetrack.yaml"),
    "botsort":   os.path.join(TRACKER_DIR, "botsort.yaml"),
}

VOTE_FRAMES = 5
MAX_VOTE_ATTEMPTS = 15

OBB_CONF_THRESH = 0.65
OBB_MIN_AREA    = 1000
OBB_MAX_AREA    = 200_000

CLS_CONF_THRESH   = 0.7
CLS_CONF_HIGH     = 0.90
CONSENSUS_RATIO   = 0.6

WARP_W, WARP_H = 224, 224

HILO_VALUES = {
    "2": +1, "3": +1, "4": +1, "5": +1, "6": +1,
    "7":  0, "8":  0, "9":  0,
    "10": -1, "J": -1, "Q": -1, "K": -1, "A": -1,
}

COLOR_BOX      = (0, 255, 0)
COLOR_UNSURE   = (0, 165, 255)
COLOR_LABEL    = (255, 255, 255)
COLOR_BG       = (0, 0, 0)

# ──────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────
def order_corners(pts: np.ndarray) -> np.ndarray:
    pts = pts.astype(np.float32).reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    d = np.diff(pts, axis=1).ravel()
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]

    return rect

def polygon_area(pts: np.ndarray) -> float:
    pts = pts.reshape(4, 2)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

def warp_card(frame: np.ndarray, corners: np.ndarray, w: int = WARP_W, h: int = WARP_H) -> np.ndarray:
    ordered = order_corners(corners)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(ordered, dst)
    return cv2.warpPerspective(frame, M, (w, h))

# ──────────────────────────────────────────────────────────────────────
# Voting & Drawing helpers
# ──────────────────────────────────────────────────────────────────────
def _weighted_winner(votes: list[tuple[str, float]]) -> tuple[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for name, w in votes:
        totals[name] += w
    grand = sum(totals.values()) or 1.0
    winner = max(totals, key=totals.get)
    return winner, totals[winner] / grand

def draw_obb(frame: np.ndarray, corners: np.ndarray, color: tuple[int, ...] = COLOR_BOX) -> None:
    pts = corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

def draw_label(frame: np.ndarray, corners: np.ndarray, label: str) -> None:
    top_idx = int(corners[:, 1].argmin())
    x, y = int(corners[top_idx, 0]), int(corners[top_idx, 1])

    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)

    cv2.rectangle(frame, (x, y - th - baseline - 6), (x + tw + 4, y), COLOR_BG, cv2.FILLED)
    cv2.putText(frame, label, (x + 2, y - baseline - 4), font, scale, COLOR_LABEL, thickness)


def draw_cached_overlay(frame: np.ndarray,
                        overlay: list[tuple[np.ndarray, tuple, str | None]]) -> None:
    """Redraw boxes + labels from the previous detection frame."""
    for corners, color, label in overlay:
        draw_obb(frame, corners, color=color)
        if label:
            draw_label(frame, corners, label)


# ──────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────
def run_pipeline(source: str, count_cards: bool = False,
                 tracker: str = "bytetrack",
                 detect_every_override: int | None = None) -> None:
    # ── Verify Apple Silicon GPU is available ────────────────────────
    compute_device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[INFO] Using compute device: {compute_device.upper()}")

    print(f"[INFO] Loading OBB model : {OBB_MODEL_PATH}")
    obb_model = YOLO(OBB_MODEL_PATH)

    print(f"[INFO] Loading CLS model : {CLS_MODEL_PATH}")
    cls_model = YOLO(CLS_MODEL_PATH)

    tracker_yaml = TRACKER_CONFIGS.get(tracker, TRACKER_CONFIGS["bytetrack"])

    with open(tracker_yaml) as f:
        tracker_cfg = yaml.safe_load(f)
    detect_every_n = detect_every_override or tracker_cfg.get("detect_every_n", 1)

    print(f"[INFO] Tracker          : {tracker} ({tracker_yaml})")
    print(f"[INFO] Detect every     : {detect_every_n} frame(s)")

    # ── Open video source ────────────────────────────────────────────
    is_webcam = source.lower() in ("webcam", "0")
    if is_webcam:
        cap = CameraStream(0, backend=cv2.CAP_AVFOUNDATION)
        delay = 1
        print("[INFO] Using threaded webcam feed")
    else:
        cap = cv2.VideoCapture(source)
        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = max(1, int(1000 / fps)) if fps > 0 else 30
        print(f"[INFO] Playing video: {source}  ({fps:.1f} FPS)")

    if not cap.isOpened():
        print("[ERROR] Cannot open video source.")
        return

    classified_cards: dict[int, str] = {}
    vote_buffer: dict[int, list[tuple[str, float]]] = defaultdict(list)
    attempt_count: dict[int, int] = defaultdict(int)
    counted_tids: set[int] = set()
    running_count: int = 0

    fps_timer = time.time()
    fps_frame_count = 0
    display_fps = 0.0

    frame_counter = 0
    cached_overlay: list[tuple[np.ndarray, tuple, str | None]] = []

    window_name = "Card Tracker"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print("[INFO] Pipeline running — press 'q' to quit, 'r' to reset\n")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            if not is_webcam:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        fps_frame_count += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 0.5:
            display_fps = fps_frame_count / elapsed
            fps_frame_count = 0
            fps_timer = time.time()

        run_detection = (frame_counter % detect_every_n == 0)
        frame_counter += 1

        if not run_detection:
            # ── Skip frame: redraw cached boxes on the fresh image ──
            draw_cached_overlay(frame, cached_overlay)

        else:
            # ── 1. OBB Tracking ─────────────────────────────────────
            results = obb_model.track(
                frame,
                persist=True,
                imgsz=640,
                conf=OBB_CONF_THRESH,
                iou=0.3,
                tracker=tracker_yaml,
                device=compute_device,
                half=True,
                verbose=False,
            )

            cached_overlay = []

            if not results or results[0].obb is None or results[0].obb.id is None:
                cv2.imshow(window_name, frame)
                if cv2.waitKey(delay) & 0xFF == ord('q'):
                    break
                continue

            obb_data = results[0].obb
            track_ids = obb_data.id.int().cpu().tolist()
            all_corners = obb_data.xyxyxyxy.cpu().numpy()

            # ── 2. Batch Collection Phase ────────────────────────────
            warped_crops = []
            active_tids = []

            for idx, (tid, corners) in enumerate(zip(track_ids, all_corners)):
                corners = corners.reshape(4, 2)

                if polygon_area(corners) < OBB_MIN_AREA or polygon_area(corners) > OBB_MAX_AREA:
                    continue

                if tid in classified_cards:
                    label = f"#{tid} {classified_cards[tid]}"
                    draw_obb(frame, corners, color=COLOR_BOX)
                    draw_label(frame, corners, label)
                    cached_overlay.append((corners.copy(), COLOR_BOX, label))
                    continue

                draw_obb(frame, corners, color=COLOR_UNSURE)
                attempt_count[tid] += 1

                if attempt_count[tid] > MAX_VOTE_ATTEMPTS and tid in vote_buffer:
                    votes = vote_buffer[tid]
                    if votes:
                        winner, w_sum = _weighted_winner(votes)
                        classified_cards[tid] = winner
                        del vote_buffer[tid]
                        if count_cards and tid not in counted_tids:
                            running_count += HILO_VALUES.get(winner, 0)
                            counted_tids.add(tid)
                    else:
                        del vote_buffer[tid]
                    cached_overlay.append((corners.copy(), COLOR_UNSURE, None))
                    continue

                warped_crops.append(warp_card(frame, corners))
                active_tids.append(tid)

                progress_label = None
                votes = vote_buffer[tid]
                if votes:
                    tentative, _ = _weighted_winner(votes)
                    progress_label = f"#{tid} ({tentative}?) {len(votes)}/{VOTE_FRAMES}"
                    draw_label(frame, corners, progress_label)

                cached_overlay.append((corners.copy(), COLOR_UNSURE, progress_label))

            # ── 3. Batched Inference Phase ───────────────────────────
            if warped_crops:
                cls_results = cls_model.predict(
                    warped_crops,
                    imgsz=224,
                    device=compute_device,
                    half=True,
                    verbose=False,
                )

                for tid, res in zip(active_tids, cls_results):
                    if res.probs is not None:
                        top_conf = float(res.probs.top1conf)
                        pred_name = cls_model.names[int(res.probs.top1)]

                        if top_conf >= CLS_CONF_THRESH:
                            weight = 2.0 if top_conf >= CLS_CONF_HIGH else 1.0
                            vote_buffer[tid].append((pred_name, weight))

                    votes = vote_buffer[tid]
                    if len(votes) >= VOTE_FRAMES:
                        winner, ratio = _weighted_winner(votes)

                        if ratio >= CONSENSUS_RATIO:
                            classified_cards[tid] = winner
                            del vote_buffer[tid]
                            if count_cards and tid not in counted_tids:
                                running_count += HILO_VALUES.get(winner, 0)
                                counted_tids.add(tid)
                        else:
                            vote_buffer[tid] = votes[len(votes) // 3:]

        # ── HUD ──────────────────────────────────────────────────────
        hud_y = 30
        fps_text = f"FPS: {display_fps:.1f}"
        fps_color = (0, 255, 0) if display_fps >= 25 else (0, 165, 255) if display_fps >= 15 else (0, 0, 255)
        (fw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(frame, fps_text, (frame.shape[1] - fw - 10, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)

        cv2.putText(frame, f"Tracked cards: {len(classified_cards)}", (10, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if count_cards:
            count_color = (0, 255, 0) if running_count > 0 else (0, 0, 255) if running_count < 0 else (0, 255, 255)
            cv2.putText(frame, f"Running Count: {running_count:+d}", (10, hud_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, count_color, 2)
            hud_y += 30

        for i, (tid, cls_name) in enumerate(sorted(classified_cards.items())):
            hilo_tag = f" ({HILO_VALUES.get(cls_name, 0):+d})" if count_cards and HILO_VALUES.get(cls_name, 0) != 0 else (" ( 0)" if count_cards else "")
            cv2.putText(frame, f"#{tid}: {cls_name}{hilo_tag}", (10, hud_y + 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        cv2.imshow(window_name, frame)

        # ── Controls ─────────────────────────────────────────────────
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'): break
        elif key == ord('r'):
            classified_cards.clear(); vote_buffer.clear(); attempt_count.clear(); counted_tids.clear(); running_count = 0
            print("[INFO] Reset — all classifications cleared.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="webcam")
    parser.add_argument("--tracker", choices=list(TRACKER_CONFIGS.keys()),
                        default="bytetrack",
                        help="Tracking algorithm: sort (fastest), bytetrack (balanced), botsort (most accurate)")
    parser.add_argument("--detect-every", type=int, default=None,
                        help="Run OBB detector every Nth frame (overrides tracker YAML value)")
    parser.add_argument("--vote-frames", type=int, default=VOTE_FRAMES)
    parser.add_argument("--obb-conf", type=float, default=OBB_CONF_THRESH)
    parser.add_argument("--cls-conf", type=float, default=CLS_CONF_THRESH)
    parser.add_argument("--consensus", type=float, default=CONSENSUS_RATIO)
    parser.add_argument("--count-cards", action="store_true", default=False)
    args = parser.parse_args()

    VOTE_FRAMES, OBB_CONF_THRESH, CLS_CONF_THRESH, CONSENSUS_RATIO = args.vote_frames, args.obb_conf, args.cls_conf, args.consensus
    run_pipeline(args.source, count_cards=args.count_cards,
                 tracker=args.tracker, detect_every_override=args.detect_every)