"""YOLOv5-Pose (PPU) Demo — DeepX SDK
----------------------------------------
Description: YOLOv5-Pose compiled with the DeepX PPU. Each row of the
             (N, 256) byte buffer encodes one person:
                 bytes   0:16  -> 4 floats : raw box (cx, cy, w, h)
                 bytes  16:20  -> 4 uint8  : (gY, gX, anchor_idx, layer_idx)
                 bytes  20:24  -> 1 float  : score
                 bytes  28:232 -> 17 (x, y, conf) floats : keypoints
             Compared to the YOLOv5 PPU detector this model uses 4 strides
             (8, 16, 32, 64) with their own anchor priors. The keypoint
             confidence channel comes out as a logit; we sigmoid it.
Input      : webcam / rpicam / video file / image file
Output     : OpenCV window with person boxes and a 17-joint skeleton.

Usage:
    python yolov5pose_ppu_demo.py                            # uses config.py
    python yolov5pose_ppu_demo.py --source webcam
    python yolov5pose_ppu_demo.py --model yolov5pose_ppu.dxnn
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

EXPECTED_CHANNELS = 256
NUM_KEYPOINTS = 17
PERSON_LABELS = ["person"]
STRIDES = np.array([8, 16, 32, 64], dtype=np.float32)

# YOLOv5-Pose ships with these per-stride anchor (w, h) priors.
ANCHORS_BY_STRIDE = {
    8:  np.array([[19, 27], [44, 40], [38, 94]], dtype=np.float32),
    16: np.array([[96, 68], [86, 152], [180, 137]], dtype=np.float32),
    32: np.array([[140, 301], [303, 264], [238, 542]], dtype=np.float32),
    64: np.array([[436, 615], [739, 380], [925, 792]], dtype=np.float32),
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class YOLOv5PosePPUDetector:
    """Decode the YOLOv5-Pose PPU buffer into people + 17-joint skeletons."""

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
        if not output_tensors or output_tensors[0].ndim != 3:
            return self._empty()
        rows = output_tensors[0][0]
        if rows.shape[1] != EXPECTED_CHANNELS:
            print(f"[WARN] Unexpected PPU channel count: {rows.shape[1]}")
            return self._empty()

        boxes_raw = rows[:, :16].view(np.float32).reshape(-1, 4)
        grid_info = rows[:, 16:20].view(np.uint8)
        scores = rows[:, 20:24].view(np.float32).flatten()
        kps_raw = rows[:, 28:232].view(np.float32).reshape(-1, NUM_KEYPOINTS, 3)

        gy = grid_info[:, 0].astype(np.float32)
        gx = grid_info[:, 1].astype(np.float32)
        anchor_idx = grid_info[:, 2]
        layer_idx = grid_info[:, 3]
        stride = STRIDES[layer_idx]

        anchor_w, anchor_h = self._lookup_anchors(stride, anchor_idx)

        cx = (boxes_raw[:, 0] * 2.0 - 0.5 + gx) * stride
        cy = (boxes_raw[:, 1] * 2.0 - 0.5 + gy) * stride
        w = (boxes_raw[:, 2] ** 2 * 4.0) * anchor_w
        h = (boxes_raw[:, 3] ** 2 * 4.0) * anchor_h
        boxes_xyxy = np.column_stack([cx - w * 0.5, cy - h * 0.5,
                                      cx + w * 0.5, cy + h * 0.5])

        # Keypoints share the same grid offset trick as the box centre.
        kpts = np.empty_like(kps_raw)
        kpts[:, :, 0] = (kps_raw[:, :, 0] * 2.0 - 0.5 + gx[:, None]) * stride[:, None]
        kpts[:, :, 1] = (kps_raw[:, :, 1] * 2.0 - 0.5 + gy[:, None]) * stride[:, None]
        kpts[:, :, 2] = _sigmoid(kps_raw[:, :, 2])

        keep = sdk.nms(boxes_xyxy, scores, self.conf_threshold, self.iou_threshold)
        if keep.size == 0:
            return self._empty()

        boxes = sdk.unletterbox_boxes(
            boxes_xyxy[keep], self._gain, self._pad, self._src_shape
        )
        kpts = sdk.unletterbox_points(
            kpts[keep], self._gain, self._pad, self._src_shape
        )
        class_ids = np.zeros(keep.size, dtype=np.int64)
        return boxes, scores[keep], class_ids, kpts

    @staticmethod
    def _lookup_anchors(stride: np.ndarray, anchor_idx: np.ndarray):
        anchor_w = np.zeros(len(stride), dtype=np.float32)
        anchor_h = np.zeros(len(stride), dtype=np.float32)
        for s, anchors in ANCHORS_BY_STRIDE.items():
            mask = stride == s
            if not np.any(mask):
                continue
            anchor_w[mask] = anchors[anchor_idx[mask], 0]
            anchor_h[mask] = anchors[anchor_idx[mask], 1]
        return anchor_w, anchor_h

    @staticmethod
    def _empty():
        e = np.empty((0,), dtype=np.float32)
        return (np.empty((0, 4), dtype=np.float32), e, e.astype(np.int64),
                np.empty((0, NUM_KEYPOINTS, 3), dtype=np.float32))

    def infer(self, frame_bgr: np.ndarray):
        return self.postprocess(self.engine.run([self.preprocess(frame_bgr)]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLOv5-Pose PPU Demo — DeepX SDK")
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
    model = YOLOv5PosePPUDetector(args.model, args.conf, args.iou)
    source = InputSource(args.source, path=_resolve_path(args),
                         webcam_index=config.WEBCAM_INDEX)

    def draw(frame: np.ndarray, result) -> None:
        boxes, scores, class_ids, kpts = result
        draw_detections(frame, boxes, scores, class_ids, PERSON_LABELS, config)
        draw_keypoints(frame, kpts, skeleton_connections=COCO_SKELETON)

    run_demo(model, draw, source, config.WINDOW_NAME, show_fps=config.SHOW_FPS)


if __name__ == "__main__":
    main()
