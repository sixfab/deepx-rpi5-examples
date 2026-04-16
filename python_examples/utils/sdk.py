"""Thin wrappers around the DeepX Python SDK.

These helpers live here so each demo file can stay focused on its own
postprocess logic without re-implementing model loading, version checks,
or the letterbox-resize that nearly every detector needs.
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple

import cv2
import numpy as np


def load_engine(model_path: str):
    """Load a .dxnn model and return (engine, input_height, input_width).

    On failure we print a friendly hint and exit. We *do not* raise here
    because the demos are meant to be run by beginners — a clean message
    is more useful than a stack trace they cannot interpret.
    """
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        print("        Set MODEL_PATH in config.py or pass --model <path>.")
        sys.exit(1)

    try:
        from dx_engine import InferenceEngine  # type: ignore
    except ImportError:
        print("[ERROR] The DeepX SDK Python binding `dx_engine` is not installed.")
        print("        It ships with your DeepX hardware package (dx_rt) and is")
        print("        not available on PyPI. See the SDK installation guide.")
        sys.exit(1)

    try:
        engine = InferenceEngine(model_path)
    except Exception as exc:  # SDK raises generic exceptions on bad models.
        print(f"[ERROR] Failed to load model {model_path}: {exc}")
        sys.exit(1)

    info = engine.get_input_tensors_info()
    # Models compiled by DX-COM use an NHWC layout, so shape == [1, H, W, C].
    input_h = int(info[0]["shape"][1])
    input_w = int(info[0]["shape"][2])

    print(f"[INFO] Loaded model: {model_path}")
    print(f"[INFO] Input size  : {input_w}x{input_h}")
    return engine, input_h, input_w


def load_labels(path: str, fallback: List[str]) -> List[str]:
    """Read a one-label-per-line file, falling back to a built-in list.

    The fallback exists so demos still work out-of-the-box without the
    user having to provide a labels file (most models use COCO-80 anyway).
    """
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            return [ln.strip() for ln in fh if ln.strip()]
    return list(fallback)


def letterbox(
    image: np.ndarray, target_size: Tuple[int, int]
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize while keeping aspect ratio, padding the rest with grey.

    Most YOLO-family detectors are trained on letterboxed inputs. Skipping
    this step distorts the image and tanks accuracy. The padding/scale
    factors are returned so detections can be projected back to the
    original frame after inference.

    Returns:
        padded:  HxWx3 letterboxed image.
        gain:    Single uniform scale factor that was applied.
        pad:     (top, left) padding offsets, in pixels.
    """
    target_h, target_w = target_size
    src_h, src_w = image.shape[:2]
    gain = min(target_h / src_h, target_w / src_w)

    new_w, new_h = int(round(src_w * gain)), int(round(src_h * gain))
    if (new_w, new_h) != (src_w, src_h):
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w_total = target_w - new_w
    pad_h_total = target_h - new_h
    top = pad_h_total // 2
    bottom = pad_h_total - top
    left = pad_w_total // 2
    right = pad_w_total - left

    padded = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),  # Standard YOLO grey fill.
    )
    return padded, gain, (top, left)


def unletterbox_boxes(
    boxes: np.ndarray, gain: float, pad: Tuple[int, int], src_shape: Tuple[int, int]
) -> np.ndarray:
    """Map x1y1x2y2 boxes from letterboxed model space back to the source.

    Mirrors the geometry done by `letterbox`: undo the pad, then divide by
    the scale factor, then clip into the source image bounds.
    """
    if boxes.size == 0:
        return boxes
    src_h, src_w = src_shape
    top, left = pad
    boxes[:, 0] = np.clip((boxes[:, 0] - left) / gain, 0, src_w - 1)
    boxes[:, 1] = np.clip((boxes[:, 1] - top) / gain, 0, src_h - 1)
    boxes[:, 2] = np.clip((boxes[:, 2] - left) / gain, 0, src_w - 1)
    boxes[:, 3] = np.clip((boxes[:, 3] - top) / gain, 0, src_h - 1)
    return boxes


def unletterbox_points(
    points: np.ndarray, gain: float, pad: Tuple[int, int], src_shape: Tuple[int, int]
) -> np.ndarray:
    """Map (..., {2,3}) keypoints from model space back to the source frame.

    Only the first two channels (x, y) are touched; an optional third
    visibility/confidence channel is left untouched.
    """
    if points.size == 0:
        return points
    src_h, src_w = src_shape
    top, left = pad
    points[..., 0] = np.clip((points[..., 0] - left) / gain, 0, src_w - 1)
    points[..., 1] = np.clip((points[..., 1] - top) / gain, 0, src_h - 1)
    return points


def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert (cx, cy, w, h) boxes to (x1, y1, x2, y2)."""
    if boxes.size == 0:
        return boxes
    half = boxes[:, 2:4] * 0.5
    return np.column_stack(
        [
            boxes[:, 0] - half[:, 0],
            boxes[:, 1] - half[:, 1],
            boxes[:, 0] + half[:, 0],
            boxes[:, 1] + half[:, 1],
        ]
    )


def nms(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    score_threshold: float,
    iou_threshold: float,
) -> np.ndarray:
    """Run cv2's NMS and return the indices of the boxes to keep."""
    if boxes_xyxy.size == 0:
        return np.empty((0,), dtype=np.int64)
    # cv2.dnn.NMSBoxes wants (x, y, w, h), so pre-convert.
    boxes_xywh = np.column_stack(
        [
            boxes_xyxy[:, 0],
            boxes_xyxy[:, 1],
            boxes_xyxy[:, 2] - boxes_xyxy[:, 0],
            boxes_xyxy[:, 3] - boxes_xyxy[:, 1],
        ]
    )
    idxs = cv2.dnn.NMSBoxes(
        boxes_xywh.tolist(),
        scores.tolist(),
        float(score_threshold),
        float(iou_threshold),
    )
    if len(idxs) == 0:
        return np.empty((0,), dtype=np.int64)
    return np.array(idxs).reshape(-1)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Tiny convenience wrapper to keep demo code readable."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
