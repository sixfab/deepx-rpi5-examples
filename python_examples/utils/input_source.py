"""Uniform frame-source abstraction for the demos.

The four supported source types — webcam, Raspberry Pi camera (rpicam),
video file, and single image — all expose the same `read() / release()`
contract. The demos can then loop over frames without caring where they
came from, and the user can switch input by editing config.py or passing
a CLI flag.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


class InputSource:
    """Open a frame source and yield BGR frames one at a time.

    Args:
        source_type: One of "webcam", "rpicam", "video", "image".
        path: File path. Required for "video" and "image".
        webcam_index: Device index for OpenCV. Used only for "webcam".

    The class purposely keeps ALL backend-specific quirks inside its
    constructor so that the rest of the demo code is unaware of them.
    """

    SUPPORTED = ("webcam", "rpicam", "video", "image")

    def __init__(
        self,
        source_type: str,
        path: Optional[str] = None,
        webcam_index: int = 0,
    ) -> None:
        if source_type not in self.SUPPORTED:
            raise ValueError(
                f"Unknown source_type {source_type!r}. "
                f"Expected one of {self.SUPPORTED}."
            )

        self.source_type = source_type
        self._cap: Optional[cv2.VideoCapture] = None
        self._picam = None  # picamera2 instance, lazy-loaded
        self._still_frame: Optional[np.ndarray] = None

        if source_type == "webcam":
            # OpenCV opens the device by integer index. We also shrink the
            # internal buffer so that read() returns the freshest frame
            # instead of a stale one — important for low-latency display.
            self._cap = cv2.VideoCapture(webcam_index)
            if not self._cap.isOpened():
                raise RuntimeError(
                    f"Could not open webcam index {webcam_index}. "
                    f"Try a different --source (video/image/rpicam) or check "
                    f"that no other process is using the camera."
                )
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        elif source_type == "rpicam":
            # picamera2 is the modern Raspberry Pi camera library. We import
            # it lazily so that x86 users can run the other source types
            # without having libcamera installed.
            try:
                from picamera2 import Picamera2  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "rpicam selected but picamera2 is not installed. "
                    "On Raspberry Pi OS run: sudo apt install -y python3-picamera2"
                ) from exc

            self._picam = Picamera2()
            video_config = self._picam.create_video_configuration(
                main={"format": "RGB888", "size": (1280, 720)}
            )
            self._picam.configure(video_config)
            self._picam.start()

        elif source_type == "video":
            if path is None:
                raise ValueError("source_type='video' requires path=...")
            self._cap = cv2.VideoCapture(path)
            if not self._cap.isOpened():
                raise RuntimeError(f"Could not open video file: {path}")

        elif source_type == "image":
            if path is None:
                raise ValueError("source_type='image' requires path=...")
            self._still_frame = cv2.imread(path)
            if self._still_frame is None:
                raise RuntimeError(f"Could not read image file: {path}")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Return (ok, frame_bgr). When ok is False the source is exhausted."""

        if self.source_type == "image":
            # Return a fresh copy each call so that drawing on it does not
            # accumulate overlays across iterations.
            return True, self._still_frame.copy()

        if self.source_type == "rpicam":
            # picamera2 returns RGB; convert to BGR to match OpenCV's world.
            frame_rgb = self._picam.capture_array()
            if frame_rgb is None:
                return False, None
            return True, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        ok, frame = self._cap.read()
        return ok, frame

    def release(self) -> None:
        """Release the underlying handle. Safe to call multiple times."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._picam is not None:
            self._picam.stop()
            self._picam = None
