<p align="center">
  <h1 align="center">♠️ Blackjack Card Counter</h1>
  <p align="center">
    Real-time playing card detection, classification, and Hi-Lo counting<br/>
    powered by YOLO OBB detection &middot; optimized for Apple Silicon
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/YOLO-v8-purple?logo=yolo&logoColor=white" alt="YOLO">
  <img src="https://img.shields.io/badge/Apple%20Silicon-MPS-black?logo=apple&logoColor=white" alt="Apple Silicon">
  <img src="https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv&logoColor=white" alt="OpenCV">
</p>

---

## Demo

<p align="center">
  <video src="demo.mp4" width="720" autoplay loop muted>
    Your browser does not support the video tag.
  </video>
</p>

> Cards are detected with oriented bounding boxes, classified by rank, and counted using the Hi-Lo system — all in real time.

---

## How It Works

```
                    ┌─────────────┐
   Video/Webcam ──▶ │  OBB Model  │──▶ Oriented Bounding Boxes
                    └─────────────┘
                           │
                    ┌──────▼──────┐
                    │   Tracker   │──▶ Persistent Track IDs
                    └─────────────┘
                           │
                    ┌──────▼──────┐
                    │  Classifier │──▶ Card Rank (2–A)
                    │  (batched)  │    with vote consensus
                    └─────────────┘
                           │
                    ┌──────▼──────┐
                    │  Hi-Lo      │──▶ Running Count & True Count
                    │  Counter    │
                    └─────────────┘
```

The pipeline runs three stages per frame:

| Stage | Model / Method | Details |
|-------|---------------|---------|
| **Detection** | YOLOv8 OBB | Oriented bounding boxes detect cards at any angle. Confidence and area filters remove noise. |
| **Tracking** | SimpleIOUTracker (default) | Greedy IoU matching assigns persistent IDs without Kalman-filter distortion. ByteTrack and BoT-SORT also available. |
| **Classification** | YOLOv8 Classifier | Warped card crops are classified in batched async inference on a background thread. A weighted voting system over multiple frames ensures consensus before finalizing a rank. |
| **Counting** | Hi-Lo System | Cards 2–6 = +1, 7–9 = 0, 10–A = −1. True Count adjusts for remaining decks. |

---

## Features

- **Oriented Bounding Box Detection** — handles cards at any rotation or perspective
- **Multi-frame Voting Consensus** — resists single-frame misclassifications
- **Hi-Lo Card Counting** — Running Count + True Count with configurable deck size
- **Multiple Tracker Options** — SimpleIOU (raw boxes), ByteTrack, BoT-SORT, SORT
- **Async Batched Classification** — background thread with queue-based batching keeps the main loop fast
- **Apple Silicon Native** — MPS acceleration out of the box, CPU fallback supported
- **Threaded Camera Stream** — non-blocking webcam capture for consistent frame rates
- **Built-in Recording** — save annotated output directly to `.mp4`

---

## Installation

```bash
git clone https://github.com/<your-username>/blackjack-card-counter.git
cd blackjack-card-counter

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Requirements

- Python 3.12+
- macOS with Apple Silicon recommended (MPS acceleration)
- Works on CPU for any platform

---

## Quick Start

### Webcam

```bash
python main.py --source webcam
```

### Video file

```bash
python main.py --source path/to/video.mp4
```

### With card counting

```bash
python main.py --source webcam --count-cards
```

### Record a demo

```bash
python main.py --source video.mp4 --count-cards --record demo.mp4
```

---

## CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | `webcam` | Video file path or `webcam` |
| `--tracker` | `simple` | Tracking algorithm: `simple`, `bytetrack`, `botsort`, `sort` |
| `--detect-every` | auto | Run detector every N frames (overrides tracker config) |
| `--count-cards` | off | Enable Hi-Lo card counting |
| `--num-decks` | `6` | Number of decks in the shoe |
| `--show-fps` | off | Display FPS counter overlay |
| `--record` | — | Save annotated output to `.mp4` file |
| `--obb-conf` | `0.55` | OBB detection confidence threshold |
| `--cls-conf` | `0.81` | Classification confidence threshold |
| `--consensus` | `0.75` | Weighted vote ratio needed to finalize a card |
| `--vote-frames` | `5` | Number of classification votes before decision |
| `--cls-wait` | `4` | Detection frames to skip before first classification attempt |

### Full example

```bash
python main.py \
  --source webcam \
  --tracker simple \
  --count-cards \
  --num-decks 8 \
  --show-fps \
  --record session.mp4 \
  --obb-conf 0.6 \
  --cls-conf 0.85
```

---

## Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Reset all tracked cards, votes, and counts |

---

## Project Structure

```
card_counter/
├── main.py                  # Entry point — pipeline orchestration
├── debug_obb.py             # Standalone OBB viewer (no tracking)
├── requirements.txt
│
├── src/
│   ├── config.py            # All thresholds, paths, and Hi-Lo values
│   ├── camera.py            # Threaded webcam capture
│   ├── simple_tracker.py    # IoU-based tracker (raw bounding boxes)
│   ├── classifier_worker.py # Async batched classification thread
│   ├── geometry.py          # Corner ordering, perspective warp, area
│   └── visualization.py     # OBB drawing, labels, vote display
│
├── weights/
│   ├── detector.pt          # YOLOv8 OBB detection model
│   └── classifier.pt        # YOLOv8 classification model
│
├── trackers/
│   ├── bytetrack.yaml       # ByteTrack config
│   ├── botsort.yaml         # BoT-SORT config
│   └── sort.yaml            # SORT-like config
│
├── evaluation/              # Evaluation scripts and ground-truth data
├── finetuning/              # Training notebooks for detector & classifier
└── segmentation/            # Card segmentation pipeline (data generation)
```

---

## Tracker Comparison

| Tracker | Method | Bounding Box | Speed | Best For |
|---------|--------|-------------|-------|----------|
| **`simple`** | Greedy IoU matching | Raw detector output | Fast | Static or slow-moving cards |
| **`bytetrack`** | Kalman + two-stage matching | Kalman-filtered | Medium | General tracking |
| **`botsort`** | Kalman + optical flow + ReID | Kalman-filtered | Slow | Fast motion, occlusions |
| **`sort`** | Kalman + Hungarian | Kalman-filtered | Fast | Simple scenes |

> **Note:** Kalman-based trackers may inflate bounding boxes on initial frames as the filter converges. The `simple` tracker avoids this by using raw detection geometry.

---

## Hi-Lo Counting System

| Cards | Count Value |
|-------|-------------|
| 2, 3, 4, 5, 6 | **+1** |
| 7, 8, 9 | **0** |
| 10, J, Q, K, A | **−1** |

**Running Count** = cumulative sum of all counted card values.

**True Count** = Running Count ÷ decks remaining, where decks remaining = (total cards − cards seen) ÷ 52.

A positive True Count indicates a player advantage; negative favors the house.

---

## License

This project is for educational and research purposes.
