"""DeepLabv3 Semantic Segmentation Demo — DeepX SDK
------------------------------------------------------
Description: Per-pixel class prediction trained on Cityscapes (19 classes
             such as road, building, person, car). The output tensor has
             shape (1, 19, H, W) of class logits; argmax along the channel
             dimension gives the per-pixel class id which we then colourise
             and blend over the original frame.
Input      : webcam / rpicam / video file / image file
Output     : OpenCV window with a translucent class map painted on top.

Usage:
    python deeplabv3_demo.py                                 # uses config.py
    python deeplabv3_demo.py --source image --path street.jpg
    python deeplabv3_demo.py --model deeplabv3.dxnn
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from utils import sdk
from utils.input_source import InputSource
from utils.label_sets import CITYSCAPES_PALETTE
from utils.runner import run_demo
from utils.visualizer import draw_semantic_mask


def _build_colormap(palette) -> np.ndarray:
    """Convert the (R, G, B) palette to a BGR LUT usable by OpenCV."""
    arr = np.asarray(palette, dtype=np.uint8)
    # OpenCV expects BGR, the palette is given as RGB.
    return arr[:, ::-1].copy()


class DeepLabv3Segmenter:
    """DeepLabv3: stretch-resize → engine → argmax over class channel."""

    def __init__(self, model_path: str) -> None:
        self.engine, self.input_h, self.input_w = sdk.load_engine(model_path)

    def preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        # Cityscapes models are forgiving about aspect ratio, so a plain
        # stretch resize matches the upstream reference implementation.
        rgb = sdk.bgr_to_rgb(frame_bgr)
        return cv2.resize(rgb, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)

    def postprocess(self, output_tensors) -> np.ndarray:
        # Output layout is (1, num_classes, H, W); argmax over the channel
        # axis collapses the per-class logits into one class id per pixel.
        return np.argmax(output_tensors[0][0], axis=0)

    def infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        return self.postprocess(self.engine.run([self.preprocess(frame_bgr)]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DeepLabv3 Demo — DeepX SDK")
    p.add_argument("--source", type=str, default=config.INPUT_SOURCE,
                   choices=list(InputSource.SUPPORTED))
    p.add_argument("--path", type=str, default=None)
    p.add_argument("--model", type=str, default=config.MODEL_PATH)
    # Semantic segmentation has no NMS / confidence gate, but the shared
    # launcher passes the global --conf / --iou to every demo. Accept and ignore.
    p.add_argument("--conf", type=float, default=None, help=argparse.SUPPRESS)
    p.add_argument("--iou", type=float, default=None, help=argparse.SUPPRESS)
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
    colormap = _build_colormap(CITYSCAPES_PALETTE)
    model = DeepLabv3Segmenter(args.model)
    source = InputSource(args.source, path=_resolve_path(args),
                         webcam_index=config.WEBCAM_INDEX)

    def draw(frame: np.ndarray, class_map: np.ndarray) -> None:
        draw_semantic_mask(frame, class_map, colormap)

    run_demo(model, draw, source, config.WINDOW_NAME, show_fps=config.SHOW_FPS)


if __name__ == "__main__":
    main()
