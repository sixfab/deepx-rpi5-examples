"""YOLOv5 Demo — DeepX SDK
-------------------------------
Description: Anchor-based multi-class object detection (COCO-80 by default).
Input      : webcam / rpicam / video file / image file
Output     : OpenCV window with drawn bounding boxes, labels and FPS.

YOLOv5 emits a tensor of shape (1, num_anchors, 5 + num_classes), where
the 5 = [cx, cy, w, h, objectness]. The final per-anchor confidence is
`objectness * max(class_scores)`. That is the only meaningful difference
from the YOLOv8 family — everything else (letterbox, NMS, drawing) is
identical.

Usage:
    python yolov5_demo.py                                  # uses config.py
    python yolov5_demo.py --source webcam
    python yolov5_demo.py --source video --path input.mp4
    python yolov5_demo.py --model yolov5.dxnn --conf 0.4
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from utils import sdk
from utils.input_source import InputSource
from utils.label_sets import COCO80
from utils.runner import run_demo
from utils.visualizer import draw_detections


class YOLOv5Detector:
    """Decode YOLOv5 outputs (anchor-based, with explicit objectness)."""

    def __init__(self, model_path, labels, conf_threshold, iou_threshold,
                 obj_threshold: float = 0.25):
        self.engine, self.input_h, self.input_w = sdk.load_engine(model_path)
        self.labels = labels
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        # Cheap pre-filter: drop anchors whose objectness alone is too low,
        # before doing the more expensive class-argmax. Identical to the
        # value used by the upstream YOLOv5 reference implementation.
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
        # outputs[i] = [cx, cy, w, h, obj_score, cls_0, cls_1, ...]
        outputs = np.squeeze(output_tensors[0])

        obj_scores = outputs[:, 4]
        keep_mask = obj_scores >= self.obj_threshold
        if not np.any(keep_mask):
            empty = np.empty((0,), dtype=np.float32)
            return np.empty((0, 4), dtype=np.float32), empty, empty

        filtered = outputs[keep_mask]
        cls_scores = filtered[:, 5:]
        class_ids = np.argmax(cls_scores, axis=1)
        # Final confidence = objectness × top-1 class probability.
        scores = obj_scores[keep_mask] * np.max(cls_scores, axis=1)

        boxes = sdk.cxcywh_to_xyxy(filtered[:, :4])
        keep = sdk.nms(boxes, scores, self.conf_threshold, self.iou_threshold)
        if keep.size == 0:
            empty = np.empty((0,), dtype=np.float32)
            return np.empty((0, 4), dtype=np.float32), empty, empty

        boxes = sdk.unletterbox_boxes(boxes[keep], self._gain, self._pad, self._src_shape)
        return boxes, scores[keep], class_ids[keep]

    def infer(self, frame_bgr: np.ndarray):
        input_tensor = self.preprocess(frame_bgr)
        outputs = self.engine.run([input_tensor])
        return self.postprocess(outputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv5 Demo — DeepX SDK")
    parser.add_argument("--source", type=str, default=config.INPUT_SOURCE,
                        choices=list(InputSource.SUPPORTED))
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--model", type=str, default=config.MODEL_PATH)
    parser.add_argument("--labels", type=str, default=config.LABEL_PATH)
    parser.add_argument("--conf", type=float, default=config.CONFIDENCE_THRESHOLD)
    parser.add_argument("--iou", type=float, default=config.IOU_THRESHOLD)
    return parser.parse_args()


def _resolve_path(args):
    if args.path is not None:
        return args.path
    if args.source == "video":
        return config.VIDEO_PATH
    if args.source == "image":
        return config.IMAGE_PATH
    return ""


def main() -> None:
    args = parse_args()
    labels = sdk.load_labels(args.labels, COCO80)
    model = YOLOv5Detector(args.model, labels, args.conf, args.iou)
    source = InputSource(args.source, path=_resolve_path(args),
                         webcam_index=config.WEBCAM_INDEX)

    def draw(frame, result):
        boxes, scores, class_ids = result
        draw_detections(frame, boxes, scores, class_ids, labels, config)

    run_demo(model, draw, source, config.WINDOW_NAME, show_fps=config.SHOW_FPS)


if __name__ == "__main__":
    main()
