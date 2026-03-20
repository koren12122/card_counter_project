"""Threaded camera stream for non-blocking frame capture."""

from __future__ import annotations

import threading

import cv2
import numpy as np


class CameraStream:
    """Continuously grabs frames on a background thread so the main
    loop never blocks waiting for the webcam."""

    def __init__(self, src: int = 0, backend: int | None = None,
                 width: int = 1280, height: int = 720):
        args = (src, backend) if backend is not None else (src,)
        self.cap = cv2.VideoCapture(*args)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera with source {src}")

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
