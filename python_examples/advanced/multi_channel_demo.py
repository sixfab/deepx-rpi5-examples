"""Multi-Channel Demo — DeepX SDK
--------------------------------
Runs detection on N video sources in parallel threads and composites the
results into a single N×N grid window.

Based on: MultiChannelAdapter.cpp / MultiChannelAdapter.h

Note: the shipped yolov8n.dxnn / YOLOv5s_512.dxnn are PPU-compiled models
that emit a packed (1, N, 32) byte buffer rather than the classic
(1, 84, N) float tensor. Decoding goes through ``utils.ppu`` and branches
on ``cfg.ultralytics``.

Architecture:
  - One thread per channel (port of channelLoop()). Each thread owns its
    own InputSource, letterbox state, and result-frame buffer.
  - The InferenceEngine is *shared* across threads (m_engine). An
    engine_lock serialises `engine.run(...)` calls — the NPU is a single
    device and concurrent run() calls are not safe. The lock is held for
    the engine call only; preprocess and postprocess run lock-free per
    thread on each thread's own buffers (no shared mutable state).

Config (config.yml → demos.multi_channel or demos.multi_channel_4):
  model_path  : Shared .dxnn model
  channels    : List of {source, path} dicts (one per tile)
  ultralytics : True for yolov8 head, False for yolov5 + anchors
  anchors     : Per-stride anchor table (only when ultralytics=False)

Usage:
  python advanced/multi_channel_demo.py
  python advanced/multi_channel_demo.py --grid 2     # force a 2x2 layout
  python advanced/multi_channel_demo.py --demo-key multi_channel_4
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from config_loader import get_demo_config
from utils import ppu, sdk
from utils.fps_counter import FPSCounter
from utils.input_source import InputSource
from utils.label_sets import COCO80
from utils.visualizer import draw_detections

# Fixed per-tile display size. Picked so 4 channels comfortably fit on a
# 1280×720-style monitor while leaving the composite legible on a laptop.
CHANNEL_W = 640
CHANNEL_H = 360


@dataclass
class ChannelContext:
    """Per-channel state shared between its worker thread and the main loop.

    The worker writes ``result_frame`` under ``frame_lock``; the
    compositor reads it under the same lock. Everything else on this
    object is touched only by the owning thread.
    """

    index: int
    source_type: str
    source_path: str
    frame_lock: threading.Lock = field(default_factory=threading.Lock)
    # Filled lazily with the first rendered frame. None means "nothing
    # ready yet" so the compositor can draw a placeholder.
    result_frame: Optional[np.ndarray] = None
    error: Optional[str] = None


def _load_shared_engine(model_path: str):
    """Load one InferenceEngine and expose its input dimensions."""
    engine, input_h, input_w = sdk.load_engine(model_path)
    return engine, input_h, input_w


def _postprocess(
    output_tensors,
    gain: float,
    pad: Tuple[int, int],
    src_shape: Tuple[int, int],
    conf_threshold: float,
    iou_threshold: float,
    ultralytics: bool,
    anchors,
):
    """Thread-safe PPU decode. Takes geom as args rather than self.*."""
    if ultralytics:
        return ppu.decode_yolov8_ppu(
            output_tensors, gain, pad, src_shape, conf_threshold, iou_threshold,
        )
    return ppu.decode_yolov5_ppu(
        output_tensors, gain, pad, src_shape, conf_threshold, iou_threshold,
        anchors or (),
    )


def _channel_loop(
    ctx: ChannelContext,
    engine,
    engine_lock: threading.Lock,
    input_h: int,
    input_w: int,
    labels: Sequence[str],
    conf_threshold: float,
    iou_threshold: float,
    webcam_index: int,
    stop: threading.Event,
    ultralytics: bool,
    anchors,
) -> None:
    """Port of MultiChannelAdapter::channelLoop().

    Reads frames from this channel's input, runs inference under the
    shared engine lock, draws detections, and publishes the rendered
    frame under the channel's frame_lock.
    """
    try:
        source = InputSource(
            ctx.source_type,
            path=ctx.source_path if ctx.source_type in ("video", "image") else None,
            webcam_index=webcam_index,
        )
    except Exception as exc:
        # Record the error so the main thread can render it instead of a
        # black tile, and exit cleanly without taking the process down.
        with ctx.frame_lock:
            ctx.error = str(exc)
        print(f"[ERROR] Channel {ctx.index} failed to open source: {exc}")
        return

    debug_done = False
    try:
        while not stop.is_set():
            ok, frame = source.read()
            if not ok or frame is None:
                # Either EOF on a video file or a camera hiccup — back off
                # briefly to avoid busy-spinning and try again.
                time.sleep(0.030)
                continue

            # Per-thread preprocess: letterbox is stateless (no shared
            # mutable state), so each thread can safely do its own.
            # Also: this thread owns its own ``frame`` reference from
            # source.read() so two channels can never accidentally share
            # the same buffer mid-decode.
            src_shape = frame.shape[:2]
            rgb = sdk.bgr_to_rgb(frame)
            padded, gain, pad = sdk.letterbox(rgb, (input_h, input_w))

            # The NPU is a single device — serialise every engine.run()
            # call so threads don't clobber each other's requests.
            try:
                with engine_lock:
                    outputs = engine.run([padded])
            except Exception as exc:
                print(f"[WARN] Channel {ctx.index} inference failed: {exc}")
                time.sleep(0.010)
                continue

            try:
                boxes, scores, class_ids = _postprocess(
                    outputs, gain, pad, src_shape,
                    conf_threshold, iou_threshold,
                    ultralytics, anchors,
                )
            except Exception as exc:
                print(f"[WARN] Channel {ctx.index} postprocess failed: {exc}")
                continue

            if not debug_done:
                raw_shape = outputs[0].shape if outputs else None
                print(
                    f"[DEBUG] ch{ctx.index} input={input_w}x{input_h} "
                    f"ultralytics={ultralytics} raw={raw_shape} "
                    f"detections={len(boxes)}"
                )
                debug_done = True

            draw_detections(frame, boxes, scores, class_ids, labels, config)

            # Fit the output tile to a fixed display size so the grid
            # compositor can blit without ever having to resize on the hot
            # path. We resize here where the worker has spare cycles.
            resized = cv2.resize(frame, (CHANNEL_W, CHANNEL_H))

            with ctx.frame_lock:
                ctx.result_frame = resized
    finally:
        source.release()


def _placeholder_tile(index: int, message: str) -> np.ndarray:
    """Dark tile with a label — used while a channel has no frame yet."""
    tile = np.full((CHANNEL_H, CHANNEL_W, 3), 20, dtype=np.uint8)
    cv2.putText(
        tile,
        f"Channel {index}: {message}",
        (20, CHANNEL_H // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 200, 200),
        2,
        cv2.LINE_AA,
    )
    return tile


def parse_args(cfg) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-Channel Demo — DeepX SDK")
    # This demo picks its own sources from config.yml → channels, so
    # --source / --path aren't meaningful here. We still accept them as
    # suppressed no-ops so the shared launcher can pass its standard flag
    # set without special-casing this demo.
    parser.add_argument("--source", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--path", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--model", type=str,
        default=getattr(cfg, "model_path", config.MODEL_PATH),
    )
    parser.add_argument(
        "--labels", type=str,
        default=getattr(cfg, "label_path", config.LABEL_PATH),
    )
    parser.add_argument(
        "--conf", type=float,
        default=getattr(cfg, "confidence_threshold", 0.45),
    )
    parser.add_argument(
        "--iou", type=float,
        default=getattr(cfg, "iou_threshold", 0.45),
    )
    parser.add_argument(
        "--grid", type=int, default=0,
        help="Force an NxN grid. Default: ceil(sqrt(num_channels)).",
    )
    parser.add_argument(
        "--demo-key", type=str, default="multi_channel",
        help="config.yml key to load (e.g. multi_channel_4).",
    )
    return parser.parse_args()


def main() -> None:
    # Two-pass parse: pre-read --demo-key so we know which config block
    # to load, then re-parse with cfg-derived defaults applied.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--demo-key", type=str, default="multi_channel")
    pre_args, _ = pre.parse_known_args()
    cfg = get_demo_config(pre_args.demo_key)
    args = parse_args(cfg)

    raw_channels = getattr(cfg, "channels", None) or []
    if len(raw_channels) < 1:
        print("[ERROR] multi_channel demo needs at least one channel in config.yml.")
        sys.exit(1)
    if len(raw_channels) < 2:
        print(
            "[WARN] Only one channel configured — running in single-tile mode. "
            "Add more entries under demos.multi_channel.channels for a grid."
        )

    labels = sdk.load_labels(args.labels, COCO80)

    try:
        engine, input_h, input_w = _load_shared_engine(args.model)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[ERROR] Failed to initialize model: {exc}")
        sys.exit(1)

    # Build one ChannelContext per configured channel.
    contexts: List[ChannelContext] = [
        ChannelContext(
            index=i,
            source_type=ch.get("source", "webcam"),
            source_path=ch.get("path") or "",
        )
        for i, ch in enumerate(raw_channels)
    ]

    engine_lock = threading.Lock()
    stop = threading.Event()
    webcam_index = int(getattr(cfg, "webcam_index", config.WEBCAM_INDEX))
    ultralytics = bool(getattr(cfg, "ultralytics", True))
    anchors = getattr(cfg, "anchors", None)

    print(
        f"[INFO] Model path : {args.model}  "
        f"input={input_w}x{input_h} ultralytics={ultralytics}"
    )

    threads: List[threading.Thread] = []
    for ctx in contexts:
        t = threading.Thread(
            target=_channel_loop,
            args=(
                ctx,
                engine,
                engine_lock,
                input_h,
                input_w,
                labels,
                float(args.conf),
                float(args.iou),
                webcam_index,
                stop,
                ultralytics,
                anchors,
            ),
            name=f"channel-{ctx.index}",
            daemon=True,
        )
        t.start()
        threads.append(t)

    # Grid geometry — matches MultiChannelAdapter.cpp lines ~40-50. Uses
    # ceil(sqrt(N)) so a 2-source setup becomes 2×2 (half empty), a
    # 5-source setup becomes 3×3, etc. The --grid override is there for
    # non-square preferences (e.g. forcing 2×2 on a 3-channel setup).
    n = len(contexts)
    grid_dim = args.grid if args.grid > 0 else max(1, math.ceil(math.sqrt(n)))
    composite_w = grid_dim * CHANNEL_W
    composite_h = grid_dim * CHANNEL_H
    composite = np.zeros((composite_h, composite_w, 3), dtype=np.uint8)

    fps = FPSCounter()
    window_name = getattr(cfg, "window_name", "DeepX — Multi-Channel")
    show_fps = getattr(cfg, "show_fps", True)

    print(f"[INFO] Running {n} channel(s) in a {grid_dim}x{grid_dim} grid. Press 'q' or ESC to quit.")
    try:
        while True:
            # Zero out only once per frame — overwriting every tile below
            # fills whatever isn't covered by a live channel.
            composite[:] = 0

            for ctx in contexts:
                col = ctx.index % grid_dim
                row = ctx.index // grid_dim
                if row >= grid_dim:
                    # More channels than the grid can show — skip; user
                    # can bump --grid to see them.
                    continue

                x0 = col * CHANNEL_W
                y0 = row * CHANNEL_H

                with ctx.frame_lock:
                    # Snapshot-under-lock so the worker can keep writing
                    # while we blit the previous frame out.
                    tile = ctx.result_frame
                    err = ctx.error

                if err is not None:
                    composite[y0:y0 + CHANNEL_H, x0:x0 + CHANNEL_W] = _placeholder_tile(
                        ctx.index, f"error ({err[:40]})"
                    )
                elif tile is None:
                    composite[y0:y0 + CHANNEL_H, x0:x0 + CHANNEL_W] = _placeholder_tile(
                        ctx.index, "warming up…"
                    )
                else:
                    composite[y0:y0 + CHANNEL_H, x0:x0 + CHANNEL_W] = tile
                    # Subtle per-tile label so you know which feed is which.
                    cv2.putText(
                        composite,
                        f"CH {ctx.index}",
                        (x0 + 10, y0 + 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

            fps.update()
            if show_fps:
                fps.draw(composite)

            cv2.imshow(window_name, composite)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("[INFO] Quit requested by user.")
                break

            # If every worker has exited (all sources exhausted or errored)
            # there's no point spinning the composite loop any longer.
            if not any(t.is_alive() for t in threads):
                print("[INFO] All channels finished.")
                break

    except KeyboardInterrupt:
        print("[INFO] Interrupted (Ctrl+C).")
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=2.0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
