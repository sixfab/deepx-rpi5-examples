"""YOLOv5-Face Demo — DeepX SDK
-----------------------------------
Description: Single-class face detection plus 5 facial landmarks per face
             (left-eye, right-eye, nose, left-mouth, right-mouth).
             Output schema: [cx, cy, w, h, obj,
                             lm0_x, lm0_y, lm1_x, lm1_y, lm2_x, lm2_y,
                             lm3_x, lm3_y, lm4_x, lm4_y, cls_score].
Input      : webcam / rpicam / video file / image file
Output     : OpenCV window with face boxes and landmark dots.

Usage:
    python yolov5face_demo.py                              # uses config.py
    python yolov5face_demo.py --source webcam
    python yolov5face_demo.py --model yolov5face.dxnn
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
FACE_LABELS = ["face"]


class YOLOv5FaceDetector:
    """YOLOv5-face: anchor-based face detector with 5 landmarks per face."""

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
        # Landmarks come BEFORE the class score in this layout.
        landmarks = filtered[:, 5:5 + NUM_LANDMARKS * 2].reshape(-1, NUM_LANDMARKS, 2)
        class_score = filtered[:, 5 + NUM_LANDMARKS * 2]
        scores = obj_scores[keep_mask] * class_score

        boxes = sdk.cxcywh_to_xyxy(filtered[:, :4])
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
    p = argparse.ArgumentParser(description="YOLOv5-Face Demo — DeepX SDK")
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
    model = YOLOv5FaceDetector(args.model, args.conf, args.iou)
    source = InputSource(args.source, path=_resolve_path(args),
                         webcam_index=config.WEBCAM_INDEX)

    def draw(frame: np.ndarray, result) -> None:
        boxes, scores, class_ids, landmarks = result
        draw_detections(frame, boxes, scores, class_ids, FACE_LABELS, config)
        # No skeleton — facial landmarks are rendered as standalone dots.
        draw_keypoints(frame, landmarks, skeleton_connections=())

    run_demo(model, draw, source, config.WINDOW_NAME, show_fps=config.SHOW_FPS)


if __name__ == "__main__":
    main()
