"""Trespassing Detection Demo — DeepX SDK
---------------------------------------
Detects when a person enters a user-defined forbidden zone (polygon).
Triggers a visual alarm overlay when trespassing is detected.

Based on: TrespassingAdapter.cpp

Note: the shipped yolov8n.dxnn is a PPU-compiled model that emits a
packed (1, N, 32) byte buffer rather than the classic (1, N, 85) float
tensor. Decoding goes through ``utils.ppu`` (anchor-free yolov8 head).

Config (config.yml → demos.trespassing):
  model_path          : Path to YOLO .dxnn model
  polygon             : List of [x,y] normalized points defining the forbidden zone
  confidence_threshold: Detection threshold
  target_class        : Class name to check (default: "person")
  ultralytics         : True for anchor-free yolov8 PPU decode
  input_width/height  : Sanity-check vs the model's reported input size
  anchors             : Per-stride list (only when ultralytics=False)

Usage:
  python advanced/trespassing_demo.py
  python advanced/trespassing_demo.py --source video --path my_video.mp4
  python advanced/trespassing_demo.py --conf 0.4
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

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
from utils.zone_utils import draw_polygon_overlay, point_in_polygon


class PpuDetector:
    """Run a PPU-compiled YOLO model and decode it via ``utils.ppu``.

    Branches on ``ultralytics`` to pick the right grid-decode formula:
    anchor-free (yolov8) vs anchor-based (yolov5/v7).
    """

    def __init__(
        self,
        model_path: str,
        labels: List[str],
        conf_threshold: float,
        iou_threshold: float,
        ultralytics: bool,
        anchors=None,
    ) -> None:
        self.engine, self.input_h, self.input_w = sdk.load_engine(model_path)
        self.labels = labels
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.ultralytics = ultralytics
        self.anchors = anchors

        self._gain: float = 1.0
        self._pad: Tuple[int, int] = (0, 0)
        self._src_shape: Tuple[int, int] = (0, 0)
        # First-frame debug dump so a user can see whether the PPU path is
        # actually returning rows on this model + video pair.
        self._debug_done = False

    def preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        self._src_shape = frame_bgr.shape[:2]
        rgb = sdk.bgr_to_rgb(frame_bgr)
        padded, self._gain, self._pad = sdk.letterbox(rgb, (self.input_h, self.input_w))
        return padded

    def postprocess(self, output_tensors):
        if self.ultralytics:
            return ppu.decode_yolov8_ppu(
                output_tensors, self._gain, self._pad, self._src_shape,
                self.conf_threshold, self.iou_threshold,
            )
        return ppu.decode_yolov5_ppu(
            output_tensors, self._gain, self._pad, self._src_shape,
            self.conf_threshold, self.iou_threshold, self.anchors or (),
        )

    def infer(self, frame_bgr: np.ndarray):
        input_tensor = self.preprocess(frame_bgr)
        outputs = self.engine.run([input_tensor])
        result = self.postprocess(outputs)
        if not self._debug_done:
            raw_shape = outputs[0].shape if outputs else None
            print(
                f"[DEBUG] input={self.input_w}x{self.input_h} "
                f"ultralytics={self.ultralytics} raw={raw_shape} "
                f"detections={len(result[0])}"
            )
            self._debug_done = True
        return result


# Full-frame fallback used when the user has not defined a valid polygon in
# config.yml. Shaped so every detection is "inside" and the alarm triggers —
# this makes the miswiring obvious on screen rather than silently doing nothing.
_FULL_FRAME_POLYGON = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


def parse_args(cfg) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trespassing Detection Demo — DeepX SDK")
    parser.add_argument(
        "--source", type=str,
        default=getattr(cfg, "input_source", "webcam"),
        choices=list(InputSource.SUPPORTED),
    )
    parser.add_argument("--path", type=str, default=None)
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
    return parser.parse_args()


def _resolve_path(args: argparse.Namespace, cfg) -> str:
    if args.path is not None:
        return args.path
    if args.source == "video":
        return getattr(cfg, "video_path", "") or ""
    if args.source == "image":
        return getattr(cfg, "image_path", "") or ""
    return ""


def _draw_alarm(frame: np.ndarray) -> None:
    """Red border + centered alert banner. Port of TrespassingAdapter alarm overlay.

    The C++ source used the string "ALEART" (typo). Fixed to "ALERT" here.
    OpenCV's Hershey fonts cannot render the ⚠ glyph, so we prefix with
    "!" as an ASCII stand-in for the warning symbol.
    """
    h, w = frame.shape[:2]

    # 1) Full-frame red border — thick enough to be visible after encoder
    # compression when the output is recorded.
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)

    # 2) Centered alert banner across the top of the frame.
    text = "! ALERT: TRESPASSING DETECTED"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.5
    thickness = 3
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max((w - tw) // 2, 10)
    y = th + 20

    # Dark backing rectangle so the red text stays legible over bright
    # scenes (snow, sky, etc.) where pure red can blend in.
    pad = 10
    cv2.rectangle(
        frame,
        (x - pad, y - th - pad),
        (x + tw + pad, y + baseline + pad),
        (0, 0, 0),
        cv2.FILLED,
    )
    cv2.putText(frame, text, (x, y), font, scale, (0, 0, 255), thickness, cv2.LINE_AA)


def main() -> None:
    cfg = get_demo_config("trespassing")
    args = parse_args(cfg)

    polygon = getattr(cfg, "polygon", None)
    if not polygon or len(polygon) < 3:
        print(
            "[WARN] No valid polygon defined in config.yml "
            "(need >=3 points) — defaulting to full-frame zone."
        )
        polygon = _FULL_FRAME_POLYGON

    target_class = getattr(cfg, "target_class", "person")

    labels = sdk.load_labels(args.labels, COCO80)

    # Resolve target class to an integer once, up front — avoids a string
    # comparison per detection per frame.
    try:
        target_id = labels.index(target_class)
    except ValueError:
        print(
            f"[WARN] target_class '{target_class}' not found in labels — "
            f"checking every detection instead."
        )
        target_id = None

    ultralytics = bool(getattr(cfg, "ultralytics", True))
    anchors = getattr(cfg, "anchors", None)

    try:
        model = PpuDetector(
            args.model, labels, args.conf, args.iou,
            ultralytics=ultralytics, anchors=anchors,
        )
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[ERROR] Failed to initialize model: {exc}")
        sys.exit(1)

    print(f"[INFO] Model path : {args.model}")

    source = InputSource(
        args.source,
        path=_resolve_path(args, cfg),
        webcam_index=getattr(cfg, "webcam_index", config.WEBCAM_INDEX),
    )
    fps = FPSCounter()
    window_name = getattr(cfg, "window_name", "DeepX — Trespassing")
    show_fps = getattr(cfg, "show_fps", True)

    print("[INFO] Streaming. Press 'q' or ESC in the window to quit.")
    try:
        while True:
            ok, frame = source.read()
            if not ok or frame is None:
                print("[INFO] Input source exhausted.")
                break

            try:
                boxes, scores, class_ids = model.infer(frame)
            except Exception as exc:
                print(f"[WARN] Inference failed on this frame: {exc}")
                continue

            h, w = frame.shape[:2]

            # Walk every detection and test its *bottom-center* against the
            # zone polygon — this is what TrespassingAdapter.cpp does on
            # lines 41-42. The bottom-center approximates where a person's
            # feet meet the ground, which is the sensible reference for
            # ground-plane geometry.
            alarm = False
            for box, cid in zip(boxes, class_ids):
                cid_int = int(cid)
                if target_id is not None and cid_int != target_id:
                    continue

                x1, y1, x2, y2 = box
                # Normalize to [0, 1] so the polygon comparison is resolution-agnostic.
                bx = ((x1 + x2) * 0.5) / w
                by = y2 / h

                if point_in_polygon((float(bx), float(by)), polygon):
                    alarm = True
                    break  # Port: C++ also breaks on first offender.

            # Draw order matters: zone first (under everything), then boxes
            # over the zone, then the alarm on top of the boxes.
            draw_polygon_overlay(frame, polygon, color=(0, 255, 255), alpha=0.25)
            draw_detections(frame, boxes, scores, class_ids, labels, config)

            if alarm:
                _draw_alarm(frame)

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


if __name__ == "__main__":
    main()
