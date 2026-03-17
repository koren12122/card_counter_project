# Blackjack Card Counter

Real-time playing card detection, tracking, and counting system using computer vision and YOLO object detection, optimized for Apple Silicon (MPS).

## Features

- **Card Detection & Tracking**: OBB (Oriented Bounding Box) detection with multiple tracker options (SORT, ByteTrack, BoT-SORT)
- **Card Classification**: Identifies card ranks (2-10, J, Q, K, A) with voting-based consensus
- **Card Counting**: Hi-Lo counting system implementation with running count display
- **Apple Silicon Optimized**: Native MPS (Metal Performance Shaders) acceleration
- **Real-time Performance**: Threaded camera stream and batched inference for optimal FPS

## Requirements

- Python 3.12+
- macOS with Apple Silicon (for MPS acceleration)
- Webcam or video file input

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Webcam Mode
```bash
python card_tracker.py --source webcam
```

### Video File Mode
```bash
python card_tracker.py --source path/to/video.mp4
```

### With Card Counting
```bash
python card_tracker.py --source webcam --count-cards
```

### Advanced Options
```bash
python card_tracker.py \
  --source webcam \
  --tracker bytetrack \
  --detect-every 3 \
  --count-cards \
  --obb-conf 0.7 \
  --cls-conf 0.75
```

## Controls

- **q**: Quit the application
- **r**: Reset all tracked cards and counts

## Configuration

- **Trackers**: `sort` (fastest), `bytetrack` (balanced), `botsort` (most accurate)
- **Detection Threshold**: Adjustable confidence levels for both detection and classification
- **Vote Frames**: Consensus-based classification over multiple frames

## Project Structure

```
card_counter/
├── card_tracker.py       # Main pipeline
├── weights/              # YOLO model weights
│   ├── detector.pt      # OBB detection model
│   └── med_cls.pt       # Card classification model
├── trackers/            # Tracker configuration files
├── utils/               # Utility functions
├── raw/                 # Raw video/image data
└── annotated/           # Annotated output images
```
