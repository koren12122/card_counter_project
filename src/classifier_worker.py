"""Background classifier worker for asynchronous card classification."""

from __future__ import annotations

import queue
import threading
import time

import numpy as np

from .config import (
    CLS_CONF_HIGH,
    CLS_CONF_THRESH,
    CONSENSUS_RATIO,
    HILO_VALUES,
    VOTE_FRAMES,
)
from .visualization import weighted_winner


class ClassifierWorker:
    """Runs card classification on a background thread with queue-based batching.

    Items are enqueued as (track_id, warped_crop) pairs.  The worker
    drains the queue and runs batched inference when either *batch_size*
    items have accumulated **or** *batch_interval* seconds have elapsed
    — whichever comes first.
    """

    def __init__(
        self,
        cls_model,
        device: str,
        classified_cards: dict[int, str],
        vote_buffer: dict[int, list[tuple[str, float]]],
        counted_tids: set[int],
        running_count: list[int],
        count_cards: bool,
        lock: threading.Lock,
        batch_size: int = 8,
        batch_interval: float = 0.15,
    ):
        self._cls_model = cls_model
        self._device = device
        self._classified_cards = classified_cards
        self._vote_buffer = vote_buffer
        self._counted_tids = counted_tids
        self._running_count = running_count
        self._count_cards = count_cards
        self._lock = lock
        self._batch_size = batch_size
        self._batch_interval = batch_interval

        self._queue: queue.Queue[tuple[int, np.ndarray] | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # ── public API ────────────────────────────────────────────────────

    def enqueue(self, tid: int, crop: np.ndarray) -> None:
        """Submit a warped card crop for asynchronous classification."""
        self._queue.put((tid, crop))

    def reset(self) -> None:
        """Discard all pending items in the queue (call after clearing shared state)."""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def stop(self) -> None:
        """Signal the worker to finish and wait for it to join."""
        self._queue.put(None)
        self._thread.join(timeout=3.0)

    # ── internals ─────────────────────────────────────────────────────

    def _drain_queue(self) -> list[tuple[int, np.ndarray]] | None:
        """Collect items until *batch_size* is reached or *batch_interval* elapses.

        Returns ``None`` when the stop sentinel is received.
        """
        batch: list[tuple[int, np.ndarray]] = []

        # Block until the first item arrives (or timeout)
        try:
            first = self._queue.get(timeout=self._batch_interval)
            if first is None:
                return None
            batch.append(first)
        except queue.Empty:
            return batch

        # Greedily grab more items without blocking
        while len(batch) < self._batch_size:
            try:
                item = self._queue.get_nowait()
                if item is None:
                    self._queue.put(None)  # re-enqueue so next iteration stops
                    break
                batch.append(item)
            except queue.Empty:
                break

        return batch

    def _run(self) -> None:
        """Background loop: drain queue -> batch predict -> update votes."""
        while True:
            batch = self._drain_queue()
            if batch is None:
                break
            if not batch:
                continue

            tids, crops = zip(*batch)

            cls_results = self._cls_model.predict(
                list(crops),
                imgsz=224,
                device=self._device,
                half=True,
                verbose=False,
            )

            with self._lock:
                self._process_results(tids, cls_results)

    def _process_results(self, tids, cls_results) -> None:
        """Accumulate votes and finalise cards that reach consensus (lock held)."""
        for tid, res in zip(tids, cls_results):
            if tid in self._classified_cards:
                continue

            if res.probs is not None:
                top_conf = float(res.probs.top1conf)
                pred_name = self._cls_model.names[int(res.probs.top1)]

                if top_conf >= CLS_CONF_THRESH:
                    weight = 2.0 if top_conf >= CLS_CONF_HIGH else 1.0
                    self._vote_buffer[tid].append((pred_name, weight))

            votes = self._vote_buffer[tid]
            if len(votes) >= VOTE_FRAMES:
                winner, ratio = weighted_winner(votes)

                if ratio >= CONSENSUS_RATIO:
                    self._classified_cards[tid] = winner
                    del self._vote_buffer[tid]
                    if self._count_cards and tid not in self._counted_tids:
                        self._running_count[0] += HILO_VALUES.get(winner, 0)
                        self._counted_tids.add(tid)
                else:
                    self._vote_buffer[tid] = votes[len(votes) // 3:]
