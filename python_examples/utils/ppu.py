"""PPU (Post Processing Unit) output decoders.

DeepX models compiled with the PPU on emit a single ``(1, N, 32)`` uint8
tensor. Each row is a ``DeviceBoundingBox_t`` from the SDK:

    bytes  0:16  -> 4 floats: raw box (cx, cy, w, h)
    bytes 16:20  -> 4 uint8 : (grid_y, grid_x, anchor_idx, layer_idx)
    bytes 20:24  -> 1 float : score
    bytes 24:28  -> 1 uint32: class id
    bytes 28:32  -> padding

The decode formula depends on the model head:

  * ultralytics (YOLOv8-style, anchor-free): ``x``, ``y``, ``w``, ``h`` are
    already pixel coordinates in model-input space. We just convert
    centre-format to xyxy, NMS, then unletterbox.
  * classic YOLOv5/YOLOv7 (anchor-based, with explicit objectness): the
    raw values are normalized; we recover pixel coords with
    ``(raw*2 - 0.5 + grid) * stride`` and ``(raw² * 4) * anchor``.

Both helpers return the same ``(boxes_xyxy, scores, class_ids)`` tuple
shape so callers can swap them based on ``cfg.ultralytics``.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from . import sdk

EXPECTED_CHANNELS = 32


def _empty():
    e = np.empty((0,), dtype=np.float32)
    return np.empty((0, 4), dtype=np.float32), e, e.astype(np.int64)


def _unpack_rows(output_tensors):
    """Slice the (1, N, 32) PPU buffer into its typed sub-arrays.

    Returns (boxes_raw, grid_info, scores, class_ids) or None if the
    tensor doesn't look like PPU output.
    """
    if not output_tensors or output_tensors[0].ndim != 3:
        return None
    rows = output_tensors[0][0]
    if rows.shape[1] != EXPECTED_CHANNELS or rows.shape[0] == 0:
        return None
    boxes_raw = rows[:, :16].view(np.float32).reshape(-1, 4)
    grid_info = rows[:, 16:20].view(np.uint8)
    scores = rows[:, 20:24].view(np.float32).flatten()
    class_ids = rows[:, 24:28].view(np.uint32).flatten().astype(np.int64)
    return boxes_raw, grid_info, scores, class_ids


def decode_yolov8_ppu(
    output_tensors,
    gain: float,
    pad: Tuple[int, int],
    src_shape: Tuple[int, int],
    conf_threshold: float,
    iou_threshold: float,
):
    """Decode an anchor-free (ultralytics YOLOv8) PPU output buffer.

    The PPU has already done the grid + DFL math, so x/y/w/h are pixel
    coordinates in letterboxed model space.
    """
    unpacked = _unpack_rows(output_tensors)
    if unpacked is None:
        return _empty()
    boxes_raw, _grid, scores, class_ids = unpacked

    boxes_xyxy = sdk.cxcywh_to_xyxy(boxes_raw)
    keep = sdk.nms(boxes_xyxy, scores, conf_threshold, iou_threshold)
    if keep.size == 0:
        return _empty()
    boxes = sdk.unletterbox_boxes(boxes_xyxy[keep], gain, pad, src_shape)
    return boxes, scores[keep], class_ids[keep]


def decode_yolov5_ppu(
    output_tensors,
    gain: float,
    pad: Tuple[int, int],
    src_shape: Tuple[int, int],
    conf_threshold: float,
    iou_threshold: float,
    anchors: Sequence[dict],
):
    """Decode an anchor-based YOLOv5/YOLOv7 PPU buffer.

    ``anchors`` is the list parsed from ``cfg.anchors`` — one dict per
    stride with ``widths`` and ``heights`` (3 entries each, one per
    anchor index). Mirrors the C++ ``YoloPostProcess::decodePpuOutputs``.
    """
    unpacked = _unpack_rows(output_tensors)
    if unpacked is None:
        return _empty()
    boxes_raw, grid_info, scores, class_ids = unpacked

    gy = grid_info[:, 0].astype(np.float32)
    gx = grid_info[:, 1].astype(np.float32)
    anchor_idx = grid_info[:, 2].astype(np.int64)
    layer_idx = grid_info[:, 3].astype(np.int64)

    # Build per-row stride / anchor_w / anchor_h vectors from the layer
    # table. Done once per frame; the inner per-row work stays vectorized.
    n = len(scores)
    stride_arr = np.zeros(n, dtype=np.float32)
    anchor_w = np.zeros(n, dtype=np.float32)
    anchor_h = np.zeros(n, dtype=np.float32)
    for li, layer in enumerate(anchors):
        mask = layer_idx == li
        if not np.any(mask):
            continue
        stride_arr[mask] = float(layer["stride"])
        widths = np.asarray(layer["widths"], dtype=np.float32)
        heights = np.asarray(layer["heights"], dtype=np.float32)
        ai = anchor_idx[mask]
        # Clamp anchor index defensively in case the PPU emits an out-of-
        # range index for a bogus row — better than indexing past the end.
        ai = np.clip(ai, 0, len(widths) - 1)
        anchor_w[mask] = widths[ai]
        anchor_h[mask] = heights[ai]

    cx = (boxes_raw[:, 0] * 2.0 - 0.5 + gx) * stride_arr
    cy = (boxes_raw[:, 1] * 2.0 - 0.5 + gy) * stride_arr
    w = (boxes_raw[:, 2] ** 2 * 4.0) * anchor_w
    h = (boxes_raw[:, 3] ** 2 * 4.0) * anchor_h

    boxes_xyxy = np.column_stack(
        [cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5]
    )

    keep = sdk.nms(boxes_xyxy, scores, conf_threshold, iou_threshold)
    if keep.size == 0:
        return _empty()
    boxes = sdk.unletterbox_boxes(boxes_xyxy[keep], gain, pad, src_shape)
    return boxes, scores[keep], class_ids[keep]
