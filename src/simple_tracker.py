"""Lightweight IoU-based tracker that preserves raw detection bounding boxes.

Unlike Kalman-based trackers (ByteTrack, BoTSORT), this tracker never modifies
the detection geometry — it only assigns persistent IDs via greedy IoU matching.
"""

from __future__ import annotations

import numpy as np


class SimpleIOUTracker:
    """Assigns persistent track IDs to OBB detections using axis-aligned IoU."""

    def __init__(self, iou_thresh: float = 0.3, max_lost: int = 30):
        self._next_id = 1
        self._tracks: dict[int, np.ndarray] = {}
        self._lost_count: dict[int, int] = {}
        self._iou_thresh = iou_thresh
        self._max_lost = max_lost

    def update(
        self, detections: list[np.ndarray]
    ) -> list[tuple[int, np.ndarray]]:
        """Match detections to existing tracks and return (tid, corners) pairs.

        Args:
            detections: list of (4, 2) corner arrays (raw OBB detections).

        Returns:
            List of (track_id, corners) for every matched + new detection.
        """
        if not self._tracks:
            return self._init_tracks(detections)

        track_ids = list(self._tracks.keys())
        track_corners = [self._tracks[tid] for tid in track_ids]

        n_det = len(detections)
        n_trk = len(track_ids)
        iou_matrix = np.zeros((n_det, n_trk), dtype=np.float32)
        for d in range(n_det):
            for t in range(n_trk):
                iou_matrix[d, t] = _aabb_iou(detections[d], track_corners[t])

        matched_dets: set[int] = set()
        matched_trks: set[int] = set()
        results: list[tuple[int, np.ndarray]] = []

        while True:
            if iou_matrix.size == 0:
                break
            flat = np.argmax(iou_matrix)
            d, t = int(flat // n_trk), int(flat % n_trk)
            if iou_matrix[d, t] < self._iou_thresh:
                break

            tid = track_ids[t]
            self._tracks[tid] = detections[d].copy()
            self._lost_count[tid] = 0
            results.append((tid, detections[d]))
            matched_dets.add(d)
            matched_trks.add(t)
            iou_matrix[d, :] = -1
            iou_matrix[:, t] = -1

        for d, corners in enumerate(detections):
            if d not in matched_dets:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = corners.copy()
                self._lost_count[tid] = 0
                results.append((tid, corners))

        stale = []
        for t, tid in enumerate(track_ids):
            if t not in matched_trks:
                self._lost_count[tid] = self._lost_count.get(tid, 0) + 1
                if self._lost_count[tid] > self._max_lost:
                    stale.append(tid)
        for tid in stale:
            del self._tracks[tid]
            del self._lost_count[tid]

        return results

    def reset(self) -> None:
        self._tracks.clear()
        self._lost_count.clear()
        self._next_id = 1

    def _init_tracks(
        self, detections: list[np.ndarray]
    ) -> list[tuple[int, np.ndarray]]:
        results: list[tuple[int, np.ndarray]] = []
        for corners in detections:
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = corners.copy()
            self._lost_count[tid] = 0
            results.append((tid, corners))
        return results


def _aabb_iou(corners1: np.ndarray, corners2: np.ndarray) -> float:
    """IoU between axis-aligned bounding boxes of two OBB corner sets."""
    x1_min, y1_min = corners1.min(axis=0)
    x1_max, y1_max = corners1.max(axis=0)
    x2_min, y2_min = corners2.min(axis=0)
    x2_max, y2_max = corners2.max(axis=0)

    inter_w = max(0.0, min(x1_max, x2_max) - max(x1_min, x2_min))
    inter_h = max(0.0, min(y1_max, y2_max) - max(y1_min, y2_min))
    inter_area = inter_w * inter_h

    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - inter_area

    return inter_area / union if union > 0 else 0.0
