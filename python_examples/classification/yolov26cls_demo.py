"""YOLOv26-Cls Classification Demo — DeepX SDK
---------------------------------------------------
Description: Single-label image classification on ImageNet-1k.
             The YOLOv26-cls head emits softmaxed probabilities directly,
             so postprocess is just an argsort.
Input      : webcam / rpicam / video file / image file
Output     : OpenCV window with the top-5 predicted labels overlaid.

Usage:
    python yolov26cls_demo.py                              # uses config.py
    python yolov26cls_demo.py --source image --path dog.jpg
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from config_loader import get_demo_config
from utils import sdk
from utils.input_source import InputSource
from utils.label_sets import imagenet1000
from utils.runner import run_demo
from utils.visualizer import draw_text_lines

TOP_K = 5


class YOLOv26ClsClassifier:
    """Resize → engine → top-K over already-softmaxed probabilities."""

    def __init__(self, model_path: str, labels: List[str]) -> None:
        self.engine, self.input_h, self.input_w = sdk.load_engine(model_path)
        self.labels = labels

    def preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = sdk.bgr_to_rgb(frame_bgr)
        return cv2.resize(rgb, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)

    def postprocess(self, output_tensors) -> List[Tuple[int, float]]:
        probs = output_tensors[0].flatten()
        top = np.argsort(probs)[::-1][:TOP_K]
        return [(int(idx), float(probs[idx])) for idx in top]

    def infer(self, frame_bgr: np.ndarray):
        return self.postprocess(self.engine.run([self.preprocess(frame_bgr)]))


def parse_args(cfg) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLOv26-Cls Demo — DeepX SDK")
    p.add_argument("--source", type=str,
                   default=getattr(cfg, "input_source", config.INPUT_SOURCE),
                   choices=list(InputSource.SUPPORTED))
    p.add_argument("--path", type=str, default=None)
    p.add_argument("--model", type=str,
                   default=getattr(cfg, "model_path", config.MODEL_PATH))
    p.add_argument("--labels", type=str,
                   default=getattr(cfg, "label_path", "") or "")
    # Classification has no NMS / confidence gate, but the shared launcher
    # passes the global --conf / --iou to every demo. Accept and ignore.
    p.add_argument("--conf", type=float, default=None, help=argparse.SUPPRESS)
    p.add_argument("--iou", type=float, default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def _resolve_path(args: argparse.Namespace, cfg) -> str:
    if args.path is not None:
        return args.path
    if args.source == "video":
        return getattr(cfg, "video_path", config.VIDEO_PATH)
    if args.source == "image":
        return getattr(cfg, "image_path", config.IMAGE_PATH)
    return ""


def _format_lines(predictions, labels):
    lines = []
    for class_id, prob in predictions:
        name = labels[class_id] if 0 <= class_id < len(labels) else f"class_{class_id}"
        lines.append(f"{prob * 100:5.1f}%  {name}")
    return lines


def main() -> None:
    cfg = get_demo_config("yolov26cls")
    args = parse_args(cfg)
    labels = sdk.load_labels(args.labels, imagenet1000())
    model = YOLOv26ClsClassifier(args.model, labels)
    source = InputSource(args.source, path=_resolve_path(args, cfg),
                         webcam_index=getattr(cfg, "webcam_index", config.WEBCAM_INDEX))

    def draw(frame: np.ndarray, predictions) -> None:
        draw_text_lines(frame, _format_lines(predictions, labels))

    window_name = getattr(cfg, "window_name", config.WINDOW_NAME)
    show_fps = getattr(cfg, "show_fps", config.SHOW_FPS)
    run_demo(model, draw, source, window_name, show_fps=show_fps)


if __name__ == "__main__":
    main()
