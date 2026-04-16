"""Rolling-average FPS counter with a one-line draw helper."""

from __future__ import annotations

import time
from collections import deque
from typing import Deque

import cv2
import numpy as np


class FPSCounter:
    """Compute a stable frames-per-second value over a sliding window.

    A naive `1 / dt` reading flickers wildly when dt is tiny. We keep the
    last `window` frame intervals and divide their count by their sum,
    which yields a smooth number suitable for an on-screen overlay.
    """

    def __init__(self, window: int = 30) -> None:
        # `deque(maxlen=...)` automatically drops the oldest entry once
        # the window is full, so we never have to trim manually.
        self._intervals: Deque[float] = deque(maxlen=window)
        self._last_ts: float = time.perf_counter()

    def update(self) -> None:
        """Mark a frame as just rendered. Call once per loop iteration."""
        now = time.perf_counter()
        self._intervals.append(now - self._last_ts)
        self._last_ts = now

    def get_fps(self) -> float:
        """Return the current rolling FPS, or 0.0 before any update."""
        if not self._intervals:
            return 0.0
        total = sum(self._intervals)
        return len(self._intervals) / total if total > 0 else 0.0

    def draw(self, frame: np.ndarray) -> None:
        """Overlay 'FPS: <value>' onto the top-left of `frame` in place."""
        text = f"FPS: {self.get_fps():.1f}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

        # Dark background rectangle keeps the text readable on light frames.
        pad = 6
        x, y = 10, 10 + th
        cv2.rectangle(
            frame,
            (x - pad, y - th - pad),
            (x + tw + pad, y + baseline),
            (0, 0, 0),
            cv2.FILLED,
        )
        cv2.putText(
            frame, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA
        )
