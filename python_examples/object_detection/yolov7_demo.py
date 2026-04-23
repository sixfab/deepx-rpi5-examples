"""YOLOv7 Demo — DeepX SDK
-------------------------------
Description: Anchor-based multi-class object detection (COCO-80 by default).
             YOLOv7 inherits the YOLOv5 output schema, so the postprocess
             code is intentionally identical — only the model weights and
             the recommended thresholds differ in practice.
Input      : webcam / rpicam / video file / image file
Output     : OpenCV window with drawn bounding boxes, labels and FPS.

Usage:
    python yolov7_demo.py                                  # uses config.py
    python yolov7_demo.py --source webcam
    python yolov7_demo.py --source video --path input.mp4
    python yolov7_demo.py --model yolov7.dxnn
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
from utils.label_sets import COCO80
from utils.runner import run_demo
from utils.visualizer import draw_detections


class YOLOv7Detector:
    """YOLOv7 head: same encoding as YOLOv5 — objectness × class scores."""

    def __init__(self, model_path, labels, conf_threshold, iou_threshold,
                 obj_threshold: float = 0.25):
        self.engine, self.input_h, self.input_w = sdk.load_engine(model_path)
        self.labels = labels
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.obj_threshold = obj_threshold

        self._gain = 1.0
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
            empty = np.empty((0,), dtype=np.float32)
            return np.empty((0, 4), dtype=np.float32), empty, empty

        filtered = outputs[keep_mask]
        cls_scores = filtered[:, 5:]
        class_ids = np.argmax(cls_scores, axis=1)
        scores = obj_scores[keep_mask] * np.max(cls_scores, axis=1)

        boxes = sdk.cxcywh_to_xyxy(filtered[:, :4])
        keep = sdk.nms(boxes, scores, self.conf_threshold, self.iou_threshold)
        if keep.size == 0:
            empty = np.empty((0,), dtype=np.float32)
            return np.empty((0, 4), dtype=np.float32), empty, empty

        boxes = sdk.unletterbox_boxes(boxes[keep], self._gain, self._pad, self._src_shape)
        return boxes, scores[keep], class_ids[keep]

    def infer(self, frame_bgr: np.ndarray):
        return self.postprocess(self.engine.run([self.preprocess(frame_bgr)]))


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv7 Demo — DeepX SDK")
    p.add_argument("--source", type=str, default=config.INPUT_SOURCE,
                   choices=list(InputSource.SUPPORTED))
    p.add_argument("--path", type=str, default=None)
    p.add_argument("--model", type=str, default=config.MODEL_PATH)
    p.add_argument("--labels", type=str, default=config.LABEL_PATH)
    p.add_argument("--conf", type=float, default=config.CONFIDENCE_THRESHOLD)
    p.add_argument("--iou", type=float, default=config.IOU_THRESHOLD)
    return p.parse_args()


def _resolve_path(args):
    if args.path is not None:
        return args.path
    return config.VIDEO_PATH if args.source == "video" else (
        config.IMAGE_PATH if args.source == "image" else "")


def main():
    args = parse_args()
    labels = sdk.load_labels(args.labels, COCO80)
    model = YOLOv7Detector(args.model, labels, args.conf, args.iou)
    source = InputSource(args.source, path=_resolve_path(args),
                         webcam_index=config.WEBCAM_INDEX)

    def draw(frame, result):
        boxes, scores, class_ids = result
        draw_detections(frame, boxes, scores, class_ids, labels, config)

    run_demo(model, draw, source, config.WINDOW_NAME, show_fps=config.SHOW_FPS)


if __name__ == "__main__":
    main()
