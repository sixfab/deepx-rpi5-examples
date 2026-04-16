"""YOLOv26 Demo — DeepX SDK
-------------------------------
Description: NMS-free object detection (COCO-80 by default).
             Like YOLOv10, the YOLOv26 head emits already-decoded boxes
             of the form [x1, y1, x2, y2, score, class_id], so postprocess
             only needs a confidence-threshold filter.
Input      : webcam / rpicam / video file / image file
Output     : OpenCV window with drawn bounding boxes, labels and FPS.

Usage:
    python yolov26_demo.py                                 # uses config.py
    python yolov26_demo.py --source webcam
    python yolov26_demo.py --model yolov26.dxnn
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


class YOLOv26Detector:
    """Decode the already-NMSed YOLOv26 output."""

    def __init__(self, model_path: str, labels: List[str],
                 conf_threshold: float, iou_threshold: float) -> None:
        del iou_threshold  # End-to-end model: NMS is folded into the head.
        self.engine, self.input_h, self.input_w = sdk.load_engine(model_path)
        self.labels = labels
        self.conf_threshold = conf_threshold

        self._gain: float = 1.0
        self._pad: Tuple[int, int] = (0, 0)
        self._src_shape: Tuple[int, int] = (0, 0)

    def preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        self._src_shape = frame_bgr.shape[:2]
        rgb = sdk.bgr_to_rgb(frame_bgr)
        padded, self._gain, self._pad = sdk.letterbox(rgb, (self.input_h, self.input_w))
        return padded

    def postprocess(self, output_tensors) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        outputs = np.squeeze(output_tensors[0])

        scores = outputs[:, 4]
        keep = scores >= self.conf_threshold
        if not np.any(keep):
            empty = np.empty((0,), dtype=np.float32)
            return np.empty((0, 4), dtype=np.float32), empty, empty

        kept = outputs[keep]
        boxes = kept[:, :4].astype(np.float32, copy=True)
        boxes = sdk.unletterbox_boxes(boxes, self._gain, self._pad, self._src_shape)
        return boxes, kept[:, 4].astype(np.float32), kept[:, 5].astype(np.int64)

    def infer(self, frame_bgr: np.ndarray):
        return self.postprocess(self.engine.run([self.preprocess(frame_bgr)]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLOv26 Demo — DeepX SDK")
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
    labels = sdk.load_labels(args.labels, COCO80)
    model = YOLOv26Detector(args.model, labels, args.conf, args.iou)
    source = InputSource(args.source, path=_resolve_path(args),
                         webcam_index=config.WEBCAM_INDEX)

    def draw(frame: np.ndarray, result) -> None:
        boxes, scores, class_ids = result
        draw_detections(frame, boxes, scores, class_ids, labels, config)

    run_demo(model, draw, source, config.WINDOW_NAME, show_fps=config.SHOW_FPS)


if __name__ == "__main__":
    main()
