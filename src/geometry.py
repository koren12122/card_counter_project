"""Geometry utilities for card detection and warping."""

from __future__ import annotations

import cv2
import numpy as np


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order corners: top-left, top-right, bottom-right, bottom-left."""
    pts = pts.astype(np.float32).reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    d = np.diff(pts, axis=1).ravel()
    rect[1] = pts[np.argmin(d)]  # top-right
    rect[3] = pts[np.argmax(d)]  # bottom-left

    return rect


def polygon_area(pts: np.ndarray) -> float:
    """Calculate area of a polygon using the shoelace formula."""
    pts = pts.reshape(4, 2)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def warp_card(frame: np.ndarray, corners: np.ndarray, w: int, h: int) -> np.ndarray:
    """Apply perspective transform to extract card region."""
    ordered = order_corners(corners)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(ordered, dst)
    return cv2.warpPerspective(frame, M, (w, h))
