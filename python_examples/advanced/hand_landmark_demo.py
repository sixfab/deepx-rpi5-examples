"""Hand Landmark Demo — DeepX SDK
--------------------------------
Detects hands and their 21 keypoint landmarks, drawing them as red dots
directly on the frame.

Based on: HandLandmarkAdapter.cpp / HandLandmarkAdapter.h

Execution model: ASYNC (RunAsync / Wait on a single thread).

Config (config.yml → demos.hand_landmark):
  model_path                     : Hand-landmark .dxnn model
  kpt_count                      : Landmarks per detection (default: 21)
  landmark_visibility_threshold  : Min visibility to draw (default: 0.5)
  input_width, input_height      : Model input dimensions
  confidence_threshold, iou_threshold

Usage:
  python advanced/hand_landmark_demo.py
  python advanced/hand_landmark_demo.py --source webcam
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from config_loader import get_demo_config
from utils import sdk
from utils.fps_counter import FPSCounter
from utils.input_source import InputSource


# Defaults match the Google MediaPipe-style hand-landmark convention.
_DEFAULT_KPT_COUNT = 21
_DEFAULT_VIS_THRESHOLD = 0.5


def _preprocess(
    frame_bgr: np.ndarray, input_w: int, input_h: int
) -> Tuple[np.ndarray, float, int, int]:
    """Letterbox resize that matches HandLandmarkAdapter::preprocess().

    Port of HandLandmarkAdapter.cpp lines 54-71. We keep this helper
    separate from utils.sdk.letterbox because this demo needs ``scale``
    and integer pad offsets (not the tuple returned by sdk.letterbox), so
    that landmark remapping can stay a direct translation of the C++ math.

    Returns:
        padded_rgb : HxWx3 RGB tensor sized (input_h, input_w)
        scale      : Uniform scale factor applied to the source frame
        pad_x      : Left padding in pixels of the letterboxed image
        pad_y      : Top  padding in pixels of the letterboxed image
    """
    h, w = frame_bgr.shape[:2]
    scale = min(input_w / w, input_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    pad_x = (input_w - new_w) // 2
    pad_y = (input_h - new_h) // 2

    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(
        resized,
        pad_y,
        input_h - new_h - pad_y,
        pad_x,
        input_w - new_w - pad_x,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return cv2.cvtColor(padded, cv2.COLOR_BGR2RGB), scale, pad_x, pad_y


def _remap_landmarks(
    keypoints: np.ndarray,
    scale: float,
    pad_x: int,
    pad_y: int,
) -> np.ndarray:
    """Project landmarks from letterboxed pixel space back to source pixels.

    The YOLOv8-pose head emits each landmark as ``(x, y)`` already in
    letterboxed input-image pixel coordinates (same space as the boxes),
    so we just undo the letterbox: subtract pad, divide by gain — the
    same transform ``sdk.unletterbox_boxes`` applies to the boxes.

    ``keypoints`` shape: (N, K, 2) or (N, K, 3). Only x/y are remapped;
    the optional visibility channel (if present) is untouched.
    """
    if keypoints.size == 0:
        return keypoints

    out = keypoints.copy().astype(np.float32)
    out[..., 0] = (out[..., 0] - pad_x) / scale
    out[..., 1] = (out[..., 1] - pad_y) / scale
    return out


class HandLandmarkModel:
    """ASYNC wrapper: RunAsync / Wait to overlap host + NPU work.

    HandLandmarkAdapter uses RunAsync rather than the blocking Run; even
    on a single thread that gives the caller a chance to do other work
    (grab the next frame, draw the previous result) while the NPU runs.
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float,
        iou_threshold: float,
        input_w: int,
        input_h: int,
        kpt_count: int,
    ) -> None:
        self.engine, model_h, model_w = sdk.load_engine(model_path)
        # Config-supplied dimensions override whatever the model header
        # reports — this lets the user retain the intended 640×640 even
        # if they swap models mid-session.
        self.input_w = int(input_w or model_w)
        self.input_h = int(input_h or model_h)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.kpt_count = int(kpt_count)

    def infer(self, frame_bgr: np.ndarray):
        padded, scale, pad_x, pad_y = _preprocess(
            frame_bgr, self.input_w, self.input_h
        )

        # RunAsync + Wait. On a single thread this is equivalent to Run,
        # but it matches the C++ adapter's call sequence exactly and keeps
        # the door open for overlapping work with the compositor thread
        # later.
        req_id = self.engine.run_async([padded])
        outputs = self.engine.wait(req_id)

        return self._postprocess(outputs, scale, pad_x, pad_y, frame_bgr.shape[:2])

    def _postprocess(
        self,
        output_tensors,
        scale: float,
        pad_x: int,
        pad_y: int,
        src_shape: Tuple[int, int],
    ):
        """Decode boxes and landmarks, then remap to source pixel space.

        Output layout (pose-style): per-anchor row is
            [cx, cy, w, h, obj, kpt0_x, kpt0_y, kpt0_v, ..., kptK-1_x, kptK-1_y, kptK-1_v]
        The ``obj`` slot is a single-class confidence (``hand``). Some
        variants omit the visibility channel — we detect that at runtime
        by looking at the trailing tensor width.
        """
        outputs = np.squeeze(output_tensors[0])

        # YOLOv8-pose exports (1, 4+1+K*3, N); we transpose to (N, ...).
        # YOLOv5-pose exports (1, N, 4+1+K*3); already flat after squeeze.
        # Detect orientation by number of rows vs. expected channels.
        expected_per_row = 5 + self.kpt_count * 3
        if outputs.ndim == 2 and outputs.shape[0] == expected_per_row:
            outputs = outputs.T  # model emitted (C, N) — flip to (N, C).
        elif outputs.ndim == 2 and outputs.shape[0] == 5 + self.kpt_count * 2:
            outputs = outputs.T

        # Whether the landmark triplet carries a visibility channel.
        per_row = outputs.shape[1] if outputs.ndim == 2 else 0
        has_visibility = per_row >= 5 + self.kpt_count * 3
        stride = 3 if has_visibility else 2

        scores = outputs[:, 4]
        keep_mask = scores >= self.conf_threshold
        if not np.any(keep_mask):
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), \
                np.empty((0, self.kpt_count, 3 if has_visibility else 2), dtype=np.float32)

        filtered = outputs[keep_mask]
        boxes = sdk.cxcywh_to_xyxy(filtered[:, :4])
        sc = filtered[:, 4]

        keep = sdk.nms(boxes, sc, self.conf_threshold, self.iou_threshold)
        if keep.size == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), \
                np.empty((0, self.kpt_count, 3 if has_visibility else 2), dtype=np.float32)

        boxes = boxes[keep]
        sc = sc[keep]

        # Box coordinates are in letterboxed pixel space; un-map to source.
        # Using the same (gain, pad) convention sdk.unletterbox_boxes uses:
        # top, left = pad_y, pad_x (note argument ordering).
        boxes = sdk.unletterbox_boxes(boxes, scale, (pad_y, pad_x), src_shape)

        # Landmarks: slice the trailing columns, reshape to (N, K, stride).
        kpt_flat = filtered[keep, 5:5 + self.kpt_count * stride]
        kpts = kpt_flat.reshape(-1, self.kpt_count, stride)
        kpts = _remap_landmarks(kpts, scale, pad_x, pad_y)

        return boxes, sc, kpts


def _draw_hands(
    frame: np.ndarray,
    boxes: np.ndarray,
    keypoints: np.ndarray,
    vis_threshold: float,
) -> None:
    """Draw each hand's bounding box + landmark dots.

    Port of HandLandmarkAdapter.cpp lines 92-98 ("Draw large red dots
    directly"). The C++ source explicitly *clears* connections after
    drawing so there are no skeleton lines — we honour that here.
    """
    for box in boxes:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    if keypoints.size == 0:
        return

    has_visibility = keypoints.shape[-1] >= 3
    for hand in keypoints:
        for kp in hand:
            # If the model reports per-landmark visibility, gate the draw
            # on it. Otherwise assume every landmark is valid.
            if has_visibility and kp[2] < vis_threshold:
                continue
            cv2.circle(
                frame,
                (int(kp[0]), int(kp[1])),
                5,
                (0, 0, 255),
                -1,
            )


def parse_args(cfg) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hand Landmark Demo — DeepX SDK")
    parser.add_argument(
        "--source", type=str,
        default=getattr(cfg, "input_source", "webcam"),
        choices=list(InputSource.SUPPORTED),
    )
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument(
        "--model", type=str,
        default=getattr(cfg, "model_path", config.MODEL_PATH),
    )
    # Accepted but unused — hand-landmark models only have one class
    # ("hand") and don't consult an external labels file. Present so the
    # shared launcher can pass --labels without a special case.
    parser.add_argument("--labels", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--conf", type=float,
        default=getattr(cfg, "confidence_threshold", 0.3),
    )
    parser.add_argument(
        "--iou", type=float,
        default=getattr(cfg, "iou_threshold", 0.4),
    )
    return parser.parse_args()


def _resolve_path(args: argparse.Namespace, cfg) -> str:
    if args.path is not None:
        return args.path
    if args.source == "video":
        return getattr(cfg, "video_path", "") or ""
    if args.source == "image":
        return getattr(cfg, "image_path", "") or ""
    return ""


def main() -> None:
    cfg = get_demo_config("hand_landmark")
    args = parse_args(cfg)

    kpt_count = int(getattr(cfg, "kpt_count", _DEFAULT_KPT_COUNT))
    if kpt_count != _DEFAULT_KPT_COUNT:
        print(f"[INFO] Using non-default kpt_count={kpt_count}.")

    vis_threshold = float(
        getattr(cfg, "landmark_visibility_threshold", _DEFAULT_VIS_THRESHOLD)
    )
    input_w = int(getattr(cfg, "input_width", 640))
    input_h = int(getattr(cfg, "input_height", 640))

    try:
        model = HandLandmarkModel(
            args.model, args.conf, args.iou, input_w, input_h, kpt_count
        )
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[ERROR] Failed to initialize model: {exc}")
        sys.exit(1)

    source = InputSource(
        args.source,
        path=_resolve_path(args, cfg),
        webcam_index=getattr(cfg, "webcam_index", config.WEBCAM_INDEX),
    )
    fps = FPSCounter()
    window_name = getattr(cfg, "window_name", "DeepX — Hand Landmarks")
    show_fps = getattr(cfg, "show_fps", True)

    print("[INFO] Streaming. Press 'q' or ESC in the window to quit.")
    try:
        while True:
            ok, frame = source.read()
            if not ok or frame is None:
                print("[INFO] Input source exhausted.")
                break

            try:
                boxes, scores, keypoints = model.infer(frame)
            except Exception as exc:
                print(f"[WARN] Inference failed on this frame: {exc}")
                continue

            _draw_hands(frame, boxes, keypoints, vis_threshold)

            fps.update()
            if show_fps:
                fps.draw(frame)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("[INFO] Quit requested by user.")
                break

    except KeyboardInterrupt:
        print("[INFO] Interrupted (Ctrl+C).")
    finally:
        source.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
