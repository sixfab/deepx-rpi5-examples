"""YOLOv8 Demo — DeepX SDK
-------------------------------
Description: Real-time multi-class object detection (COCO-80 by default).
Input      : webcam / rpicam / video file / image file
Output     : OpenCV window with drawn bounding boxes, labels and FPS.

YOLOv8 emits a single tensor of shape (1, 4 + num_classes, num_anchors).
Unlike YOLOv5/v7, there is NO objectness channel — the per-class score
itself is the confidence. That changes how postprocess() filters boxes,
but the overall flow (decode → NMS → draw) stays identical.

Usage:
    python yolov8_demo.py                                  # uses config.py
    python yolov8_demo.py --source webcam
    python yolov8_demo.py --source video --path input.mp4
    python yolov8_demo.py --source image --path photo.jpg
    python yolov8_demo.py --model yolov8.dxnn --labels labels.txt
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


class YOLOv8Detector:
    """Loads a YOLOv8 .dxnn model and runs detection on one frame at a time."""

    def __init__(
        self,
        model_path: str,
        labels: List[str],
        conf_threshold: float,
        iou_threshold: float,
    ) -> None:
        self.engine, self.input_h, self.input_w = sdk.load_engine(model_path)
        self.labels = labels
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Cached per-frame so postprocess can map results back to original space.
        self._gain: float = 1.0
        self._pad: Tuple[int, int] = (0, 0)
        self._src_shape: Tuple[int, int] = (0, 0)

    # ----- preprocess --------------------------------------------------
    def preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Convert BGR frame into the letterboxed RGB tensor the model wants."""
        self._src_shape = frame_bgr.shape[:2]
        rgb = sdk.bgr_to_rgb(frame_bgr)
        padded, self._gain, self._pad = sdk.letterbox(rgb, (self.input_h, self.input_w))
        return padded

    # ----- postprocess -------------------------------------------------
    def postprocess(self, output_tensors) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode model outputs into (boxes_xyxy, scores, class_ids).

        The raw tensor is (1, 4+C, N). We transpose to (N, 4+C) so each
        row is one anchor: [cx, cy, w, h, score_class0, score_class1, ...].
        """
        outputs = np.transpose(np.squeeze(output_tensors[0]))  # (N, 4 + C)

        # Per-anchor: pick the strongest class and use its score as the
        # confidence. No multiplication by an objectness term — YOLOv8
        # folds objectness into the class scores during training.
        cls_scores = outputs[:, 4:]
        class_ids = np.argmax(cls_scores, axis=1)
        scores = np.max(cls_scores, axis=1)

        boxes = sdk.cxcywh_to_xyxy(outputs[:, :4])
        keep = sdk.nms(boxes, scores, self.conf_threshold, self.iou_threshold)
        if keep.size == 0:
            empty = np.empty((0,), dtype=np.float32)
            return np.empty((0, 4), dtype=np.float32), empty, empty

        boxes = boxes[keep]
        # Map detections from letterboxed model space back to source frame.
        boxes = sdk.unletterbox_boxes(boxes, self._gain, self._pad, self._src_shape)
        return boxes, scores[keep], class_ids[keep]

    # ----- public --------------------------------------------------------
    def infer(self, frame_bgr: np.ndarray):
        input_tensor = self.preprocess(frame_bgr)
        outputs = self.engine.run([input_tensor])
        return self.postprocess(outputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 Demo — DeepX SDK")
    parser.add_argument("--source", type=str, default=config.INPUT_SOURCE,
                        choices=list(InputSource.SUPPORTED))
    parser.add_argument("--path", type=str, default=None,
                        help="Path to video or image file (overrides config).")
    parser.add_argument("--model", type=str, default=config.MODEL_PATH)
    parser.add_argument("--labels", type=str, default=config.LABEL_PATH)
    parser.add_argument("--conf", type=float, default=config.CONFIDENCE_THRESHOLD)
    parser.add_argument("--iou", type=float, default=config.IOU_THRESHOLD)
    return parser.parse_args()


def _resolve_path(args: argparse.Namespace) -> str:
    """Pick the right file path for video/image based on CLI + config."""
    if args.path is not None:
        return args.path
    if args.source == "video":
        return config.VIDEO_PATH
    if args.source == "image":
        return config.IMAGE_PATH
    return ""  # webcam/rpicam don't need a path.


def main() -> None:
    args = parse_args()
    labels = sdk.load_labels(args.labels, COCO80)

    model = YOLOv8Detector(args.model, labels, args.conf, args.iou)
    source = InputSource(args.source, path=_resolve_path(args),
                         webcam_index=config.WEBCAM_INDEX)

    def draw(frame: np.ndarray, result) -> None:
        boxes, scores, class_ids = result
        draw_detections(frame, boxes, scores, class_ids, labels, config)

    run_demo(model, draw, source, config.WINDOW_NAME, show_fps=config.SHOW_FPS)


if __name__ == "__main__":
    main()
