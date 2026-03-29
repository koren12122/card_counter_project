"""
Card Tracking & Classification Pipeline (Apple Silicon Optimized)

Main entry point for real-time card detection, tracking, and classification.
"""

from __future__ import annotations

import argparse
import threading
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

from src.camera import CameraStream
from src.classifier_worker import ClassifierWorker
from src.config import (
    CLS_BATCH_INTERVAL,
    CLS_BATCH_SIZE,
    CLS_CONF_THRESH,
    CLS_MODEL_PATH,
    CLS_WAIT_FRAMES,
    COLOR_BOX,
    COLOR_UNSURE,
    CONSENSUS_RATIO,
    HILO_VALUES,
    MAX_VOTE_ATTEMPTS,
    OBB_CONF_THRESH,
    OBB_MAX_AREA,
    OBB_MIN_AREA,
    OBB_MODEL_PATH,
    TRACKER_CONFIGS,
    VOTE_FRAMES,
    WARP_H,
    WARP_W,
)
from src.geometry import polygon_area, warp_card
from src.simple_tracker import SimpleIOUTracker
from src.visualization import (
    draw_cached_overlay,
    draw_label,
    draw_obb,
    weighted_winner,
)



def run_pipeline(source: str, count_cards: bool = False,
                 tracker: str = "simple",
                 detect_every_override: int | None = None,
                 show_fps: bool = False,
                 num_decks: int = 6,
                 record: str | None = None) -> None:
    """Run the card tracking and classification pipeline."""
    # ── Verify Apple Silicon GPU is available ────────────────────────
    compute_device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[INFO] Using compute device: {compute_device.upper()}")

    print(f"[INFO] Loading OBB model : {OBB_MODEL_PATH}")
    obb_model = YOLO(OBB_MODEL_PATH)

    print(f"[INFO] Loading CLS model : {CLS_MODEL_PATH}")
    cls_model = YOLO(CLS_MODEL_PATH)

    use_simple_tracker = (tracker == "simple")
    tracker_yaml = TRACKER_CONFIGS.get(tracker, TRACKER_CONFIGS["bytetrack"])

    with open(tracker_yaml) as f:
        tracker_cfg = yaml.safe_load(f)
    detect_every_n = detect_every_override or tracker_cfg.get("detect_every_n", 1)

    iou_tracker: SimpleIOUTracker | None = None
    if use_simple_tracker:
        iou_tracker = SimpleIOUTracker(iou_thresh=0.3, max_lost=detect_every_n * 15)

    tracker_label = "SimpleIOUTracker (raw OBB boxes)" if use_simple_tracker else f"{tracker} ({tracker_yaml})"
    print(f"[INFO] Tracker          : {tracker_label}")
    print(f"[INFO] Detect every     : {detect_every_n} frame(s)")
    print(f"[INFO] CLS wait        : {CLS_WAIT_FRAMES} detection frame(s) "
          f"(~{CLS_WAIT_FRAMES * detect_every_n} raw frames)")

    # ── Open video source ────────────────────────────────────────────
    is_webcam = source.lower() in ("webcam", "0")
    if is_webcam:
        try:
            cap = CameraStream(0, backend=cv2.CAP_AVFOUNDATION)
            delay = 1
            print("[INFO] Using threaded webcam feed")
            print("[INFO] Initializing camera...")
            time.sleep(2.0)
        except RuntimeError as e:
            print(f"[ERROR] {e}")
            print("[ERROR] Try checking camera permissions or if another app is using the camera")
            return
    else:
        cap = cv2.VideoCapture(source)
        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = max(1, int(1000 / fps)) if fps > 0 else 30
        print(f"[INFO] Playing video: {source}  ({fps:.1f} FPS)")

        if not cap.isOpened():
            print("[ERROR] Cannot open video source.")
            return

    # ── Recording setup ────────────────────────────────────────────
    video_writer: cv2.VideoWriter | None = None
    if record:
        rec_fps = fps if not is_webcam and fps > 0 else 30.0
        print(f"[INFO] Recording to     : {record} ({rec_fps:.1f} FPS)")

    lock = threading.Lock()
    classified_cards: dict[int, str] = {}
    vote_buffer: dict[int, list[tuple[str, float]]] = defaultdict(list)
    attempt_count: dict[int, int] = defaultdict(int)
    counted_tids: set[int] = set()
    running_count: list[int] = [0]

    fps_timer = time.time()
    fps_frame_count = 0
    display_fps = 0.0

    frame_counter = 0
    cached_overlay: list[tuple[np.ndarray, tuple, str | None]] = []

    window_name = "Card Tracker"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    # Verify we can read at least one frame
    if is_webcam:
        test_ret, test_frame = cap.read()
        if not test_ret or test_frame is None:
            print("[ERROR] Cannot read frames from camera. Check camera permissions.")
            cap.release()
            return
        print(f"[INFO] Camera ready - Frame size: {test_frame.shape[1]}x{test_frame.shape[0]}")

    print("[INFO] Pipeline running — press 'q' to quit, 'r' to reset\n")

    worker = ClassifierWorker(
        cls_model=cls_model,
        device=compute_device,
        classified_cards=classified_cards,
        vote_buffer=vote_buffer,
        counted_tids=counted_tids,
        running_count=running_count,
        count_cards=count_cards,
        lock=lock,
        batch_size=CLS_BATCH_SIZE,
        batch_interval=CLS_BATCH_INTERVAL,
    )

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            if not is_webcam:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            print("[ERROR] Failed to read frame from webcam.")
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
            # ── 1. OBB Detection + ID assignment ─────────────────────
            cached_overlay = []

            if use_simple_tracker:
                results = obb_model.predict(
                    frame, imgsz=640, conf=OBB_CONF_THRESH, iou=0.3,
                    device=compute_device, half=True, verbose=False,
                )
                if not results or results[0].obb is None:
                    cv2.imshow(window_name, frame)
                    if cv2.waitKey(delay) & 0xFF == ord('q'):
                        break
                    continue
                raw_corners = [
                    c.reshape(4, 2)
                    for c in results[0].obb.xyxyxyxy.cpu().numpy()
                ]
                tracked = iou_tracker.update(raw_corners)
            else:
                results = obb_model.track(
                    frame, persist=True, imgsz=640, conf=OBB_CONF_THRESH,
                    iou=0.3, tracker=tracker_yaml, device=compute_device,
                    half=True, verbose=False,
                )
                if not results or results[0].obb is None or results[0].obb.id is None:
                    cv2.imshow(window_name, frame)
                    if cv2.waitKey(delay) & 0xFF == ord('q'):
                        break
                    continue
                obb_data = results[0].obb
                track_ids = obb_data.id.int().cpu().tolist()
                all_corners = obb_data.xyxyxyxy.cpu().numpy()
                tracked = [
                    (tid, corners.reshape(4, 2))
                    for tid, corners in zip(track_ids, all_corners)
                ]

            # ── 2. Card Processing & Crop Collection ─────────────────
            crops_to_enqueue: list[tuple[int, np.ndarray]] = []

            with lock:
                for tid, corners in tracked:

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
                            winner, w_sum = weighted_winner(votes)
                            classified_cards[tid] = winner
                            del vote_buffer[tid]
                            if count_cards and tid not in counted_tids:
                                running_count[0] += HILO_VALUES.get(winner, 0)
                                counted_tids.add(tid)
                        else:
                            del vote_buffer[tid]
                        cached_overlay.append((corners.copy(), COLOR_UNSURE, None))
                        continue

                    if attempt_count[tid] < CLS_WAIT_FRAMES:
                        cached_overlay.append((corners.copy(), COLOR_UNSURE, None))
                        continue

                    crops_to_enqueue.append(
                        (tid, warp_card(frame, corners, WARP_W, WARP_H))
                    )

                    progress_label = None
                    votes = vote_buffer[tid]
                    if votes:
                        tentative, _ = weighted_winner(votes)
                        progress_label = f"#{tid} ({tentative}?) {len(votes)}/{VOTE_FRAMES}"
                        draw_label(frame, corners, progress_label)

                    cached_overlay.append((corners.copy(), COLOR_UNSURE, progress_label))

            # ── 3. Enqueue for Async Classification ──────────────────
            for tid, crop in crops_to_enqueue:
                worker.enqueue(tid, crop)

        # ── HUD ──────────────────────────────────────────────────────
        hud_y = 30
        if show_fps:
            fps_text = f"FPS: {display_fps:.1f}"
            fps_color = (0, 255, 0) if display_fps >= 25 else (0, 165, 255) if display_fps >= 15 else (0, 0, 255)
            (fw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.putText(frame, fps_text, (frame.shape[1] - fw - 10, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)

        with lock:
            num_classified = len(classified_cards)
            current_count = running_count[0]
            cards_seen = len(counted_tids)
            sorted_cards = sorted(classified_cards.items())

        cv2.putText(frame, f"Tracked cards: {num_classified}", (10, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if count_cards:
            decks_remaining = max((num_decks * 52 - cards_seen) / 52, 0.5)
            true_count = current_count / decks_remaining

            count_color = (0, 255, 0) if current_count > 0 else (0, 0, 255) if current_count < 0 else (0, 255, 255)
            tc_color = (0, 255, 0) if true_count > 0 else (0, 0, 255) if true_count < 0 else (0, 255, 255)

            cv2.putText(frame, f"Running Count: {current_count:+d}", (10, hud_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, count_color, 2)
            cv2.putText(frame, f"True Count: {true_count:+.1f}  (decks left: {decks_remaining:.1f})", (10, hud_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tc_color, 2)
            hud_y += 60

        for i, (tid, cls_name) in enumerate(sorted_cards):
            hilo_tag = f" ({HILO_VALUES.get(cls_name, 0):+d})" if count_cards and HILO_VALUES.get(cls_name, 0) != 0 else (" ( 0)" if count_cards else "")
            cv2.putText(frame, f"#{tid}: {cls_name}{hilo_tag}", (10, hud_y + 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        if record:
            if video_writer is None:
                h, w = frame.shape[:2]
                rec_fps_val = fps if not is_webcam and fps > 0 else 30.0
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(record, fourcc, rec_fps_val, (w, h))
                print(f"[INFO] Recording started: {w}x{h} @ {rec_fps_val:.1f} FPS")
            video_writer.write(frame)

        cv2.imshow(window_name, frame)

        # ── Controls ─────────────────────────────────────────────────
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            with lock:
                classified_cards.clear()
                vote_buffer.clear()
                attempt_count.clear()
                counted_tids.clear()
                running_count[0] = 0
            if iou_tracker:
                iou_tracker.reset()
            worker.reset()
            print("[INFO] Reset — all classifications cleared.")

    worker.stop()
    if video_writer is not None:
        video_writer.release()
        print(f"[INFO] Recording saved to: {record}")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="webcam")
    tracker_choices = ["simple"] + list(TRACKER_CONFIGS.keys())
    parser.add_argument("--tracker", choices=tracker_choices,
                        default="simple",
                        help="Tracking algorithm: simple (raw OBB, default), bytetrack, botsort, sort")
    parser.add_argument("--detect-every", type=int, default=None,
                        help="Run OBB detector every Nth frame (overrides tracker YAML value)")
    parser.add_argument("--vote-frames", type=int, default=VOTE_FRAMES)
    parser.add_argument("--cls-wait", type=int, default=CLS_WAIT_FRAMES,
                        help="Detection frames to skip before classifying a new card")
    parser.add_argument("--obb-conf", type=float, default=OBB_CONF_THRESH)
    parser.add_argument("--cls-conf", type=float, default=CLS_CONF_THRESH)
    parser.add_argument("--consensus", type=float, default=CONSENSUS_RATIO)
    parser.add_argument("--count-cards", action="store_true", default=False)
    parser.add_argument("--show-fps", action="store_true", default=False,
                        help="Display FPS counter on the video feed")
    parser.add_argument("--num-decks", type=int, default=6,
                        help="Number of decks in the shoe (default: 6)")
    parser.add_argument("--record", type=str, default=None,
                        help="Record output to an mp4 file (e.g. --record demo.mp4)")
    args = parser.parse_args()

    # Update global config with CLI arguments
    import src.config as config
    config.VOTE_FRAMES = args.vote_frames
    config.CLS_WAIT_FRAMES = args.cls_wait
    config.OBB_CONF_THRESH = args.obb_conf
    config.CLS_CONF_THRESH = args.cls_conf
    config.CONSENSUS_RATIO = args.consensus

    run_pipeline(args.source, count_cards=args.count_cards,
                 tracker=args.tracker, detect_every_override=args.detect_every,
                 show_fps=args.show_fps, num_decks=args.num_decks,
                 record=args.record)
