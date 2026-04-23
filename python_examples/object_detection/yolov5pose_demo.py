"""YOLOv5-Pose Demo — DeepX SDK
-----------------------------------
Description: Single-class person detection plus 17 COCO keypoints per person.
             Output schema is YOLOv5-style — [cx, cy, w, h, obj, cls,
             kpt0_x, kpt0_y, kpt0_v, ..., kpt16_x, kpt16_y, kpt16_v]
             with one class ("person").
Input      : webcam / rpicam / video file / image file
Output     : OpenCV window with bounding boxes and skeleton overlays.

Usage:
    python yolov5pose_demo.py                              # uses config.py
    python yolov5pose_demo.py --source webcam
    python yolov5pose_demo.py --model yolov5pose.dxnn
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
from utils.visualizer import COCO_SKELETON, draw_detections, draw_keypoints

NUM_KEYPOINTS = 17
PERSON_LABELS = ["person"]


class YOLOv5PoseDetector:
    """YOLOv5-pose: anchor-based person detector with per-anchor keypoints."""

    def __init__(self, model_path: str, conf_threshold: float,
                 iou_threshold: float, obj_threshold: float = 0.25) -> None:
        self.engine, self.input_h, self.input_w = sdk.load_engine(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.obj_threshold = obj_threshold

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

        obj_scores = outputs[:, 4]
        keep_mask = obj_scores >= self.obj_threshold
        if not np.any(keep_mask):
            return self._empty()

        filtered = outputs[keep_mask]
        scores = obj_scores[keep_mask] * filtered[:, 5]  # single-class confidence
        kpts = filtered[:, 6:6 + NUM_KEYPOINTS * 3].reshape(-1, NUM_KEYPOINTS, 3)

        boxes = sdk.cxcywh_to_xyxy(filtered[:, :4])
        keep = sdk.nms(boxes, scores, self.conf_threshold, self.iou_threshold)
        if keep.size == 0:
            return self._empty()

        boxes = sdk.unletterbox_boxes(boxes[keep], self._gain, self._pad, self._src_shape)
        kpts = sdk.unletterbox_points(kpts[keep], self._gain, self._pad, self._src_shape)
        class_ids = np.zeros(keep.size, dtype=np.int64)
        return boxes, scores[keep], class_ids, kpts

    @staticmethod
    def _empty():
        e = np.empty((0,), dtype=np.float32)
        return (np.empty((0, 4), dtype=np.float32), e, e,
                np.empty((0, NUM_KEYPOINTS, 3), dtype=np.float32))

    def infer(self, frame_bgr: np.ndarray):
        return self.postprocess(self.engine.run([self.preprocess(frame_bgr)]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLOv5-Pose Demo — DeepX SDK")
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
    model = YOLOv5PoseDetector(args.model, args.conf, args.iou)
    source = InputSource(args.source, path=_resolve_path(args),
                         webcam_index=config.WEBCAM_INDEX)

    def draw(frame: np.ndarray, result) -> None:
        boxes, scores, class_ids, kpts = result
        draw_detections(frame, boxes, scores, class_ids, PERSON_LABELS, config)
        draw_keypoints(frame, kpts, COCO_SKELETON)

    run_demo(model, draw, source, config.WINDOW_NAME, show_fps=config.SHOW_FPS)


if __name__ == "__main__":
    main()
