"""YOLOv26-OBB Demo — DeepX SDK
-----------------------------------
Description: Oriented bounding box (rotated rectangle) detection.
             Trained on DOTA v1 (15 classes such as plane, ship, harbor).
             The head emits already-decoded rows of the form
             [cx, cy, w, h, score, class_id, angle_rad]. We turn each
             (cx, cy, w, h, angle) tuple into the four polygon corners
             and draw them with cv2.polylines.
Input      : webcam / rpicam / video file / image file
Output     : OpenCV window with rotated boxes and class labels.

Usage:
    python yolov26obb_demo.py                              # uses config.py
    python yolov26obb_demo.py --source image --path aerial.jpg
    python yolov26obb_demo.py --model yolov26obb.dxnn
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from utils import sdk
from utils.input_source import InputSource
from utils.label_sets import DOTAV1
from utils.runner import run_demo
from utils.visualizer import draw_obb


def xywhr_to_polygons(rboxes: np.ndarray) -> np.ndarray:
    """Convert (cx, cy, w, h, angle) rows into (N, 4, 2) corner points."""
    if rboxes.size == 0:
        return np.empty((0, 4, 2), dtype=np.float32)

    ctr = rboxes[:, :2]
    w = rboxes[:, 2:3]
    h = rboxes[:, 3:4]
    angle = rboxes[:, 4:5]

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    # Two half-axis vectors aligned with the rotated rectangle.
    vec_w = np.concatenate([w * 0.5 * cos_a, w * 0.5 * sin_a], axis=-1)
    vec_h = np.concatenate([-h * 0.5 * sin_a, h * 0.5 * cos_a], axis=-1)

    return np.stack(
        [ctr + vec_w + vec_h, ctr + vec_w - vec_h,
         ctr - vec_w - vec_h, ctr - vec_w + vec_h],
        axis=1,
    ).astype(np.float32)


class YOLOv26OBBDetector:
    """YOLOv26-OBB: end-to-end rotated detector (NMS already done)."""

    def __init__(self, model_path: str, conf_threshold: float,
                 iou_threshold: float) -> None:
        del iou_threshold
        self.engine, self.input_h, self.input_w = sdk.load_engine(model_path)
        self.conf_threshold = conf_threshold

        self._gain: float = 1.0
        self._pad: Tuple[int, int] = (0, 0)
        self._src_shape: Tuple[int, int] = (0, 0)

    def preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        self._src_shape = frame_bgr.shape[:2]
        rgb = sdk.bgr_to_rgb(frame_bgr)
        padded, self._gain, self._pad = sdk.letterbox(rgb, (self.input_h, self.input_w))
        return padded

    def postprocess(self, output_tensors):
        outputs = np.squeeze(output_tensors[0])

        scores = outputs[:, 4]
        keep = scores >= self.conf_threshold
        if not np.any(keep):
            empty = np.empty((0,), dtype=np.float32)
            return (np.empty((0, 4, 2), dtype=np.float32), empty, empty)

        kept = outputs[keep]
        cx = (kept[:, 0] - self._pad[1]) / self._gain
        cy = (kept[:, 1] - self._pad[0]) / self._gain
        w = kept[:, 2] / self._gain
        h = kept[:, 3] / self._gain
        rboxes = np.column_stack([cx, cy, w, h, kept[:, 6]])

        polygons = xywhr_to_polygons(rboxes)
        # Keep polygons inside the original frame for cleaner drawing.
        src_h, src_w = self._src_shape
        polygons[..., 0] = np.clip(polygons[..., 0], 0, src_w - 1)
        polygons[..., 1] = np.clip(polygons[..., 1], 0, src_h - 1)

        return polygons, kept[:, 4].astype(np.float32), kept[:, 5].astype(np.int64)

    def infer(self, frame_bgr: np.ndarray):
        return self.postprocess(self.engine.run([self.preprocess(frame_bgr)]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLOv26-OBB Demo — DeepX SDK")
    p.add_argument("--source", type=str, default=config.INPUT_SOURCE,
                   choices=list(InputSource.SUPPORTED))
    p.add_argument("--path", type=str, default=None)
    p.add_argument("--model", type=str, default=config.MODEL_PATH)
    p.add_argument("--labels", type=str, default=config.LABEL_PATH)
    p.add_argument("--conf", type=float, default=config.CONFIDENCE_THRESHOLD)
    p.add_argument("--iou", type=float, default=config.IOU_THRESHOLD)
    return p.parse_args()


def _resolve_path(args: argparse.Namespace) -> str:
    if args.path is not None:
        return args.path
    if args.source == "video":
        return config.VIDEO_PATH
    if args.source == "image":
        return config.IMAGE_PATH
    return ""


def main() -> None:
    args = parse_args()
    labels = sdk.load_labels(args.labels, DOTAV1)
    model = YOLOv26OBBDetector(args.model, args.conf, args.iou)
    source = InputSource(args.source, path=_resolve_path(args),
                         webcam_index=config.WEBCAM_INDEX)

    def draw(frame: np.ndarray, result) -> None:
        polygons, scores, class_ids = result
        draw_obb(frame, polygons, scores, class_ids, labels)

    run_demo(model, draw, source, config.WINDOW_NAME, show_fps=config.SHOW_FPS)


if __name__ == "__main__":
    main()
