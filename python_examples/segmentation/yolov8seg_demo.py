"""YOLOv8-Seg Instance Segmentation Demo — DeepX SDK
-------------------------------------------------------
Description: Per-instance segmentation on COCO-80. The network emits two
             tensors: detections of shape (1, 4 + 80 + 32, N) and a proto
             tensor of shape (1, 32, mh, mw). We pick the surviving
             detections after NMS, multiply their 32-d mask coefficients
             with the proto basis and apply sigmoid to recover one binary
             mask per detection.
Input      : webcam / rpicam / video file / image file
Output     : OpenCV window with bounding boxes blended over coloured masks.

Usage:
    python yolov8seg_demo.py                                # uses config.py
    python yolov8seg_demo.py --source image --path bus.jpg
    python yolov8seg_demo.py --model yolov8seg.dxnn
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from utils import sdk
from utils.input_source import InputSource
from utils.label_sets import COCO80
from utils.runner import run_demo
from utils.visualizer import draw_detections, draw_masks

NUM_CLASSES = 80
MASK_DIM = 32  # Number of basis vectors in the proto tensor.


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Plain sigmoid; the mask logits are tiny so no overflow guard needed."""
    return 1.0 / (1.0 + np.exp(-x))


class YOLOv8SegDetector:
    """YOLOv8 instance segmentation: detection head + mask prototypes."""

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

    def postprocess(self, output_tensors):
        # Tensor 0: detection rows. YOLOv8 stores them transposed, so we
        # flip to get (N_anchors, 4 + 80 + 32) where the trailing 32 are
        # mask coefficients per detection.
        outputs = np.transpose(np.squeeze(output_tensors[0]))

        cls_scores = outputs[:, 4:4 + NUM_CLASSES]
        scores = np.max(cls_scores, axis=1)
        class_ids = np.argmax(cls_scores, axis=1)
        mask_coefs = outputs[:, 4 + NUM_CLASSES:]

        boxes = sdk.cxcywh_to_xyxy(outputs[:, :4])
        keep = sdk.nms(boxes, scores, self.conf_threshold, self.iou_threshold)
        if keep.size == 0:
            return self._empty()

        # Proto basis is shared across all detections in a frame.
        proto = np.squeeze(output_tensors[1])
        c, mh, mw = proto.shape

        # Linear combination: (K, 32) @ (32, mh*mw) → (K, mh*mw) → (K, mh, mw).
        masks = _sigmoid(mask_coefs[keep] @ proto.reshape(c, -1))
        masks = masks.reshape(-1, mh, mw)

        masks_in_src = self._project_masks(masks, boxes[keep])
        boxes_src = sdk.unletterbox_boxes(
            boxes[keep], self._gain, self._pad, self._src_shape
        )
        return boxes_src, scores[keep], class_ids[keep], masks_in_src

    def _project_masks(self, masks: np.ndarray, boxes_model_space: np.ndarray) -> np.ndarray:
        """Resize each mask to model input, crop by box, then back to source frame."""
        src_h, src_w = self._src_shape
        # 1) Upsample low-res masks to the model input resolution.
        scaled = np.empty((len(masks), self.input_h, self.input_w), dtype=np.float32)
        for i, mask in enumerate(masks):
            scaled[i] = cv2.resize(
                mask, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR
            )

        # 2) Zero everything outside each detection's box so masks don't bleed.
        for i, box in enumerate(boxes_model_space):
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(self.input_w, x2), min(self.input_h, y2)
            scaled[i, :y1, :] = 0
            scaled[i, y2:, :] = 0
            scaled[i, :, :x1] = 0
            scaled[i, :, x2:] = 0

        # 3) Strip the letterbox padding and resize back to the source frame.
        top, left = self._pad
        unpad_h = int(round(src_h * self._gain))
        unpad_w = int(round(src_w * self._gain))
        cropped = scaled[:, top:top + unpad_h, left:left + unpad_w]

        out = np.empty((len(cropped), src_h, src_w), dtype=np.float32)
        for i, mask in enumerate(cropped):
            out[i] = cv2.resize(
                mask, (src_w, src_h), interpolation=cv2.INTER_LINEAR
            )
        return out

    @staticmethod
    def _empty():
        e = np.empty((0,), dtype=np.float32)
        return (
            np.empty((0, 4), dtype=np.float32),
            e,
            e.astype(np.int64),
            np.empty((0, 0, 0), dtype=np.float32),
        )

    def infer(self, frame_bgr: np.ndarray):
        return self.postprocess(self.engine.run([self.preprocess(frame_bgr)]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLOv8-Seg Demo — DeepX SDK")
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
    model = YOLOv8SegDetector(args.model, args.conf, args.iou)
    source = InputSource(args.source, path=_resolve_path(args),
                         webcam_index=config.WEBCAM_INDEX)

    def draw(frame: np.ndarray, result) -> None:
        boxes, scores, class_ids, masks = result
        # Masks first so the boxes/labels stay visible on top of the overlay.
        draw_masks(frame, masks, class_ids)
        draw_detections(frame, boxes, scores, class_ids, labels, config)

    run_demo(model, draw, source, config.WINDOW_NAME, show_fps=config.SHOW_FPS)


if __name__ == "__main__":
    main()
