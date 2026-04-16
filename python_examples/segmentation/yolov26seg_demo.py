"""YOLOv26-Seg Instance Segmentation Demo — DeepX SDK
--------------------------------------------------------
Description: Per-instance segmentation on COCO-80. Unlike YOLOv8-seg, the
             head emits already-decoded detection rows of the form
             [x1, y1, x2, y2, score, class_id, mask_coef0, ..., mask_coefM].
             That means no NMS on the host: we just threshold by score and
             use the surviving rows' mask coefficients with the proto basis.
Input      : webcam / rpicam / video file / image file
Output     : OpenCV window with bounding boxes blended over coloured masks.

Usage:
    python yolov26seg_demo.py                                # uses config.py
    python yolov26seg_demo.py --source image --path bus.jpg
    python yolov26seg_demo.py --model yolov26seg.dxnn
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

DETECTION_COLS = 6  # x1, y1, x2, y2, score, class_id


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class YOLOv26SegDetector:
    """YOLOv26 instance segmentation: end-to-end decoder + mask prototypes."""

    def __init__(self, model_path: str, conf_threshold: float,
                 iou_threshold: float) -> None:
        # YOLOv26 ships with NMS baked in, so iou_threshold is a no-op here.
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
        keep_mask = scores >= self.conf_threshold
        if not np.any(keep_mask):
            return self._empty()

        kept = outputs[keep_mask]
        boxes_model = kept[:, :4].astype(np.float32)
        scores_kept = kept[:, 4].astype(np.float32)
        class_ids = kept[:, 5].astype(np.int64)
        mask_coefs = kept[:, DETECTION_COLS:]

        # Apply the same proto basis the YOLOv8 family uses.
        proto = np.squeeze(output_tensors[1])
        c, mh, mw = proto.shape
        masks = _sigmoid(mask_coefs @ proto.reshape(c, -1))
        masks = masks.reshape(-1, mh, mw)

        masks_in_src = self._project_masks(masks, boxes_model)
        boxes_src = sdk.unletterbox_boxes(
            boxes_model.copy(), self._gain, self._pad, self._src_shape
        )
        return boxes_src, scores_kept, class_ids, masks_in_src

    def _project_masks(self, masks: np.ndarray, boxes_model_space: np.ndarray) -> np.ndarray:
        """Mirror of the YOLOv8-seg path: upsample → box-crop → unletterbox."""
        src_h, src_w = self._src_shape
        scaled = np.empty((len(masks), self.input_h, self.input_w), dtype=np.float32)
        for i, mask in enumerate(masks):
            scaled[i] = cv2.resize(
                mask, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR
            )

        for i, box in enumerate(boxes_model_space):
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(self.input_w, x2), min(self.input_h, y2)
            scaled[i, :y1, :] = 0
            scaled[i, y2:, :] = 0
            scaled[i, :, :x1] = 0
            scaled[i, :, x2:] = 0

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
    p = argparse.ArgumentParser(description="YOLOv26-Seg Demo — DeepX SDK")
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
    model = YOLOv26SegDetector(args.model, args.conf, args.iou)
    source = InputSource(args.source, path=_resolve_path(args),
                         webcam_index=config.WEBCAM_INDEX)

    def draw(frame: np.ndarray, result) -> None:
        boxes, scores, class_ids, masks = result
        draw_masks(frame, masks, class_ids)
        draw_detections(frame, boxes, scores, class_ids, labels, config)

    run_demo(model, draw, source, config.WINDOW_NAME, show_fps=config.SHOW_FPS)


if __name__ == "__main__":
    main()
