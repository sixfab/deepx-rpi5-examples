"""SCRFD Face Detection Demo — DeepX SDK
-------------------------------------------
Description: Anchor-free face detection with 5 facial landmarks per face.
             SCRFD emits one (score, bbox, kps) triplet per FPN level
             (typically strides 8, 16 and 32), with two anchors per cell.
             Boxes are stored as distances to the anchor centre, so we
             have to rebuild the grid coordinates and project them back.
Input      : webcam / rpicam / video file / image file
Output     : OpenCV window with face boxes and landmark dots.

Usage:
    python scrfd_demo.py                                   # uses config.py
    python scrfd_demo.py --source webcam
    python scrfd_demo.py --model scrfd.dxnn
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
from utils.runner import run_demo
from utils.visualizer import draw_detections, draw_keypoints

NUM_LANDMARKS = 5
ANCHORS_PER_CELL = 2
FACE_LABELS = ["face"]


class SCRFDDetector:
    """SCRFD multi-stride face detector."""

    def __init__(self, model_path: str, conf_threshold: float,
                 iou_threshold: float) -> None:
        self.engine, self.input_h, self.input_w = sdk.load_engine(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        self._gain: float = 1.0
        self._pad: Tuple[int, int] = (0, 0)
        self._src_shape: Tuple[int, int] = (0, 0)

    def preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        self._src_shape = frame_bgr.shape[:2]
        rgb = sdk.bgr_to_rgb(frame_bgr)
        padded, self._gain, self._pad = sdk.letterbox(rgb, (self.input_h, self.input_w))
        return padded

    def _bucket_outputs(self, output_tensors):
        """Group (score, bbox, kps) tensors by their anchor count.

        Each FPN level produces three tensors with the same N (number of
        anchors at that level). We tell them apart by their channel size:
        score=1, bbox=4, kps=10.
        """
        buckets: dict = {}
        for t in output_tensors:
            if t.ndim != 3 or t.shape[0] != 1:
                continue
            _, n, c = t.shape
            entry = buckets.setdefault(n, {})
            if c == 1:
                entry["score"] = t
            elif c == 4:
                entry["bbox"] = t
            elif c == NUM_LANDMARKS * 2:
                entry["kps"] = t
        return buckets

    def postprocess(self, output_tensors):
        all_boxes, all_scores, all_kps = [], [], []

        for n, parts in self._bucket_outputs(output_tensors).items():
            if {"score", "bbox", "kps"} - parts.keys():
                continue

            score = parts["score"].reshape(-1)
            bbox = parts["bbox"].reshape(-1, 4)
            kps = parts["kps"].reshape(-1, NUM_LANDMARKS * 2)

            # Recover stride from N = 2 * (input_w / stride) * (input_h / stride).
            hw = max(1, int(round(np.sqrt(n // ANCHORS_PER_CELL))))
            stride = max(1, self.input_w // hw)

            loc = np.arange(n) // ANCHORS_PER_CELL
            gx, gy = loc % hw, loc // hw
            cx, cy = gx * stride, gy * stride

            x1 = cx - bbox[:, 0] * stride
            y1 = cy - bbox[:, 1] * stride
            x2 = cx + bbox[:, 2] * stride
            y2 = cy + bbox[:, 3] * stride
            boxes = np.column_stack([x1, y1, x2, y2]).astype(np.float32)

            kx = cx[:, None] + kps[:, 0::2] * stride
            ky = cy[:, None] + kps[:, 1::2] * stride
            level_kps = np.stack([kx, ky], axis=-1).astype(np.float32)

            all_boxes.append(boxes)
            all_scores.append(score.astype(np.float32))
            all_kps.append(level_kps)

        if not all_boxes:
            return self._empty()

        boxes = np.vstack(all_boxes)
        scores = np.concatenate(all_scores)
        landmarks = np.vstack(all_kps)

        keep = sdk.nms(boxes, scores, self.conf_threshold, self.iou_threshold)
        if keep.size == 0:
            return self._empty()

        boxes = sdk.unletterbox_boxes(boxes[keep], self._gain, self._pad, self._src_shape)
        landmarks = sdk.unletterbox_points(
            landmarks[keep], self._gain, self._pad, self._src_shape
        )
        class_ids = np.zeros(keep.size, dtype=np.int64)
        return boxes, scores[keep], class_ids, landmarks

    @staticmethod
    def _empty():
        e = np.empty((0,), dtype=np.float32)
        return (np.empty((0, 4), dtype=np.float32), e, e,
                np.empty((0, NUM_LANDMARKS, 2), dtype=np.float32))

    def infer(self, frame_bgr: np.ndarray):
        return self.postprocess(self.engine.run([self.preprocess(frame_bgr)]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SCRFD Face Demo — DeepX SDK")
    p.add_argument("--source", type=str, default=config.INPUT_SOURCE,
                   choices=list(InputSource.SUPPORTED))
    p.add_argument("--path", type=str, default=None)
    p.add_argument("--model", type=str, default=config.MODEL_PATH)
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
    model = SCRFDDetector(args.model, args.conf, args.iou)
    source = InputSource(args.source, path=_resolve_path(args),
                         webcam_index=config.WEBCAM_INDEX)

    def draw(frame: np.ndarray, result) -> None:
        boxes, scores, class_ids, landmarks = result
        draw_detections(frame, boxes, scores, class_ids, FACE_LABELS, config)
        draw_keypoints(frame, landmarks, skeleton_connections=())

    run_demo(model, draw, source, config.WINDOW_NAME, show_fps=config.SHOW_FPS)


if __name__ == "__main__":
    main()
