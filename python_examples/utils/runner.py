"""Shared frame-loop driver.

Every demo follows the same lifecycle: open an input source, run the
model, draw results, update FPS, show the frame, quit on 'q'. The only
parts that differ between demos are the model class (preprocess +
inference + postprocess) and the drawing callback.

`run_demo()` captures that lifecycle in one place so each demo file can
stay short and focus on what is actually unique about it.
"""

from __future__ import annotations

from typing import Callable, Protocol

import cv2
import numpy as np

from .fps_counter import FPSCounter
from .input_source import InputSource


class _Inferable(Protocol):
    """Minimum interface that the runner needs from a model object."""

    def infer(self, frame_bgr: np.ndarray):
        """Take a BGR frame, return whatever postprocess produced."""


def run_demo(
    model: _Inferable,
    draw_callback: Callable[[np.ndarray, object], None],
    source: InputSource,
    window_name: str,
    show_fps: bool = True,
) -> None:
    """Drive the read → infer → draw → display loop until the user quits.

    Args:
        model: An object with an `.infer(frame_bgr)` method that returns
            an opaque "result" payload (whatever shape the demo decides).
        draw_callback: Function called as `draw_callback(frame, result)`
            to overlay the result on the frame in place.
        source: Already-opened InputSource.
        window_name: Title of the OpenCV display window.
        show_fps: Whether to overlay a rolling FPS counter.
    """
    fps = FPSCounter()
    print(f"[INFO] Streaming. Press 'q' or ESC in the window to quit.")

    try:
        while True:
            ok, frame = source.read()
            if not ok or frame is None:
                # Video reached EOF or camera disconnected — exit cleanly.
                print("[INFO] Input source exhausted.")
                break

            try:
                result = model.infer(frame)
            except Exception as exc:
                # Don't crash the whole loop on one bad frame; print and
                # keep going so the user can debug interactively.
                print(f"[WARN] Inference failed on this frame: {exc}")
                continue

            draw_callback(frame, result)

            fps.update()
            if show_fps:
                fps.draw(frame)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 'q' or ESC
                print("[INFO] Quit requested by user.")
                break

    except KeyboardInterrupt:
        print("[INFO] Interrupted (Ctrl+C).")
    finally:
        source.release()
        cv2.destroyAllWindows()
