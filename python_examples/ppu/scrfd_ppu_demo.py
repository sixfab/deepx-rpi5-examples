"""SCRFD (PPU) Face Detection Demo — DeepX SDK
-------------------------------------------------
Description: SCRFD compiled with the DeepX PPU. Each row of the (N, 64)
             byte buffer carries everything needed for a face detection:
                 bytes  0:16  -> 4 floats : box distances (l, t, r, b)
                 bytes 16:20  -> 4 uint8  : (gY, gX, _, layer_idx)
                 bytes 20:24  -> 1 float  : score
                 bytes 24:64  -> 5 (x, y) -> 10 floats: facial landmarks
             SCRFD is anchor-free, so unlike YOLOv5/v7 the box decoding
             uses (gx, gy) ± distances * stride directly — no anchor table.
Input      : webcam / rpicam / video file / image file
Output     : OpenCV window with face boxes and 5 landmark dots per face.

Usage:
    python scrfd_ppu_demo.py                                 # uses config.py
    python scrfd_ppu_demo.py --source webcam
    python scrfd_ppu_demo.py --model scrfd_ppu.dxnn
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

EXPECTED_CHANNELS = 64
NUM_LANDMARKS = 5
STRIDES = np.array([8, 16, 32], dtype=np.float32)
FACE_LABELS = ["face"]


class SCRFDPPUDetector:
    """Decode the 64-byte-per-row PPU buffer into faces and landmarks."""

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
        kps_raw = rows[:, 24:].view(np.float32).reshape(-1, NUM_LANDMARKS, 2)

        gy = grid_info[:, 0].astype(np.float32)
        gx = grid_info[:, 1].astype(np.float32)
        layer_idx = grid_info[:, 3]
        stride = STRIDES[layer_idx]

        # Distance-from-center decoding (anchor-free).
        boxes_xyxy = np.column_stack([
            (gx - boxes_raw[:, 0]) * stride,
            (gy - boxes_raw[:, 1]) * stride,
            (gx + boxes_raw[:, 2]) * stride,
            (gy + boxes_raw[:, 3]) * stride,
        ])

        landmarks = np.empty_like(kps_raw)
        landmarks[:, :, 0] = (kps_raw[:, :, 0] + gx[:, None]) * stride[:, None]
        landmarks[:, :, 1] = (kps_raw[:, :, 1] + gy[:, None]) * stride[:, None]

        keep = sdk.nms(boxes_xyxy, scores, self.conf_threshold, self.iou_threshold)
        if keep.size == 0:
            return self._empty()

        boxes = sdk.unletterbox_boxes(
            boxes_xyxy[keep], self._gain, self._pad, self._src_shape
        )
        landmarks = sdk.unletterbox_points(
            landmarks[keep], self._gain, self._pad, self._src_shape
        )
        class_ids = np.zeros(keep.size, dtype=np.int64)
        return boxes, scores[keep], class_ids, landmarks

    @staticmethod
    def _empty():
        e = np.empty((0,), dtype=np.float32)
        return (np.empty((0, 4), dtype=np.float32), e, e.astype(np.int64),
                np.empty((0, NUM_LANDMARKS, 2), dtype=np.float32))

    def infer(self, frame_bgr: np.ndarray):
        return self.postprocess(self.engine.run([self.preprocess(frame_bgr)]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SCRFD PPU Demo — DeepX SDK")
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
    model = SCRFDPPUDetector(args.model, args.conf, args.iou)
    source = InputSource(args.source, path=_resolve_path(args),
                         webcam_index=config.WEBCAM_INDEX)

    def draw(frame: np.ndarray, result) -> None:
        boxes, scores, class_ids, landmarks = result
        draw_detections(frame, boxes, scores, class_ids, FACE_LABELS, config)
        draw_keypoints(frame, landmarks, skeleton_connections=())

    run_demo(model, draw, source, config.WINDOW_NAME, show_fps=config.SHOW_FPS)


if __name__ == "__main__":
    main()
