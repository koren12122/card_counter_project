"""Visualization and voting utilities for card tracking."""

from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np

from .config import COLOR_BG, COLOR_BOX, COLOR_LABEL


def weighted_winner(votes: list[tuple[str, float]]) -> tuple[str, float]:
    """Compute weighted consensus from classification votes."""
    totals: dict[str, float] = defaultdict(float)
    for name, w in votes:
        totals[name] += w
    grand = sum(totals.values()) or 1.0
    winner = max(totals, key=totals.get)
    return winner, totals[winner] / grand


def draw_obb(frame: np.ndarray, corners: np.ndarray, color: tuple[int, ...] = COLOR_BOX) -> None:
    """Draw oriented bounding box on frame."""
    pts = corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)


def draw_label(frame: np.ndarray, corners: np.ndarray, label: str) -> None:
    """Draw label above the top corner of the bounding box."""
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
