"""Configuration constants for the card tracking pipeline."""

from __future__ import annotations

import os

# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OBB_MODEL_PATH = os.path.join(SCRIPT_DIR, "weights", "detector.pt")
CLS_MODEL_PATH = os.path.join(SCRIPT_DIR, "weights", "classifier.pt")

TRACKER_DIR = os.path.join(SCRIPT_DIR, "trackers")
TRACKER_CONFIGS = {
    "sort":      os.path.join(TRACKER_DIR, "sort.yaml"),
    "bytetrack": os.path.join(TRACKER_DIR, "bytetrack.yaml"),
    "botsort":   os.path.join(TRACKER_DIR, "botsort.yaml"),
}

# ──────────────────────────────────────────────────────────────────────
# Detection & Classification
# ──────────────────────────────────────────────────────────────────────
VOTE_FRAMES = 5
MAX_VOTE_ATTEMPTS = 20
CLS_WAIT_FRAMES   = 4

OBB_CONF_THRESH = 0.55
OBB_MIN_AREA    = 2000
OBB_MAX_AREA    = 200_000

CLS_CONF_THRESH   = 0.81
CLS_CONF_HIGH     = 0.95
CONSENSUS_RATIO   = 0.75

CLS_BATCH_SIZE      = 8
CLS_BATCH_INTERVAL  = 0.1

WARP_W, WARP_H = 224, 224

# ──────────────────────────────────────────────────────────────────────
# Card Counting
# ──────────────────────────────────────────────────────────────────────
HILO_VALUES = {
    "2": +1, "3": +1, "4": +1, "5": +1, "6": +1,
    "7":  0, "8":  0, "9":  0,
    "10": -1, "J": -1, "Q": -1, "K": -1, "A": -1,
}

# ──────────────────────────────────────────────────────────────────────
# Colors
# ──────────────────────────────────────────────────────────────────────
COLOR_BOX      = (0, 255, 0)
COLOR_UNSURE   = (0, 165, 255)
COLOR_LABEL    = (255, 255, 255)
COLOR_BG       = (0, 0, 0)
