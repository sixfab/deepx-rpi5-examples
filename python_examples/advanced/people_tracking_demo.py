"""People Tracking Demo — DeepX SDK
----------------------------------
Tracks people across frames using centroid-based greedy matching.
Each person gets a persistent ID displayed on their bounding box.

Based on: PeopleTrackingAdapter.cpp

Note: the shipped YOLOV5S_PPU.dxnn is a PPU-compiled model (input
512x512) that emits a packed (1, N, 32) byte buffer rather than the
classic (1, N, 85) float tensor. Decoding goes through ``utils.ppu``
with the YOLOv5 anchor table from config.yml.

Config (config.yml → demos.people_tracking):
  model_path          : Path to YOLO .dxnn model
  max_missing_frames  : Frames before track is deleted (default: 10)
  max_distance        : Normalized distance threshold for matching (default: 0.1)
  confidence_threshold
  ultralytics         : False for the YOLOv5 PPU head used here
  input_width/height  : 512x512 (must match YOLOV5S_PPU compile size)
  anchors             : Per-stride anchor table

Usage:
  python advanced/people_tracking_demo.py
  python advanced/people_tracking_demo.py --source webcam
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Sequence, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from config_loader import get_demo_config
from utils import ppu, sdk
from utils.fps_counter import FPSCounter
from utils.input_source import InputSource
from utils.label_sets import COCO80
from utils.tracker import CentroidTracker
from utils.visualizer import color_for, draw_text_lines


class PpuDetector:
    """Run a PPU-compiled YOLO model and decode it via ``utils.ppu``."""

    def __init__(
        self,
        model_path: str,
        labels: List[str],
        conf_threshold: float,
        iou_threshold: float,
        ultralytics: bool,
        anchors=None,
    ) -> None:
        self.engine, self.input_h, self.input_w = sdk.load_engine(model_path)
        self.labels = labels
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.ultralytics = ultralytics
        self.anchors = anchors

        self._gain: float = 1.0
        self._pad: Tuple[int, int] = (0, 0)
        self._src_shape: Tuple[int, int] = (0, 0)
        self._debug_done = False

    def preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        self._src_shape = frame_bgr.shape[:2]
        rgb = sdk.bgr_to_rgb(frame_bgr)
        padded, self._gain, self._pad = sdk.letterbox(rgb, (self.input_h, self.input_w))
        return padded

    def postprocess(self, output_tensors):
        if self.ultralytics:
            return ppu.decode_yolov8_ppu(
                output_tensors, self._gain, self._pad, self._src_shape,
                self.conf_threshold, self.iou_threshold,
            )
        return ppu.decode_yolov5_ppu(
            output_tensors, self._gain, self._pad, self._src_shape,
            self.conf_threshold, self.iou_threshold, self.anchors or (),
        )

    def infer(self, frame_bgr: np.ndarray):
        input_tensor = self.preprocess(frame_bgr)
        outputs = self.engine.run([input_tensor])
        result = self.postprocess(outputs)
        if not self._debug_done:
            raw_shape = outputs[0].shape if outputs else None
            print(
                f"[DEBUG] input={self.input_w}x{self.input_h} "
                f"ultralytics={self.ultralytics} raw={raw_shape} "
                f"detections={len(result[0])}"
            )
            self._debug_done = True
        return result


# COCO index 0 == "person". We still check the label string as a fallback,
# mirroring the defensive filter in PeopleTrackingAdapter.cpp line 31.
_PERSON_CLASS_ID = 0


def parse_args(cfg) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="People Tracking Demo — DeepX SDK")
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
    parser.add_argument(
        "--labels", type=str,
        default=getattr(cfg, "label_path", config.LABEL_PATH),
    )
    parser.add_argument(
        "--conf", type=float,
        default=getattr(cfg, "confidence_threshold", 0.45),
    )
    parser.add_argument(
        "--iou", type=float,
        default=getattr(cfg, "iou_threshold", 0.45),
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


def _is_person(class_id: int, labels: Sequence[str]) -> bool:
    """True if this detection is a person.

    Matches PeopleTrackingAdapter.cpp line 31 ("Filter by Class ID 0") but
    also falls back to a label-string check so non-COCO label files still
    work (e.g. custom-trained 1-class person models).
    """
    if class_id == _PERSON_CLASS_ID:
        return True
    if 0 <= class_id < len(labels):
        return labels[class_id].lower() == "person"
    return False


def _draw_person_box(
    frame: np.ndarray,
    box: np.ndarray,
    track_id: int,
) -> None:
    """Draw one tracked person's box + "Person [ID: N]" label.

    Colours are keyed off the track ID rather than the class ID so the
    same person keeps the same colour across frames — this is what makes
    a tracking demo visually readable.
    """
    x1, y1, x2, y2 = box.astype(int)
    color = color_for(track_id)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    text = f"Person [ID: {track_id}]"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)

    # Flip label above-box / inside-box so it never disappears off-screen
    # for people near the top edge — same trick draw_detections uses.
    label_y = y1 - 6 if y1 - th - 6 > 0 else y1 + th + 6
    cv2.rectangle(
        frame,
        (x1, label_y - th - 4),
        (x1 + tw + 4, label_y + 4),
        color,
        cv2.FILLED,
    )
    cv2.putText(
        frame,
        text,
        (x1 + 2, label_y),
        font,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def main() -> None:
    cfg = get_demo_config("people_tracking")
    args = parse_args(cfg)

    max_missing_frames = int(getattr(cfg, "max_missing_frames", 10))
    max_distance = float(getattr(cfg, "max_distance", 0.1))

    labels = sdk.load_labels(args.labels, COCO80)

    ultralytics = bool(getattr(cfg, "ultralytics", False))
    anchors = getattr(cfg, "anchors", None)

    try:
        model = PpuDetector(
            args.model, labels, args.conf, args.iou,
            ultralytics=ultralytics, anchors=anchors,
        )
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[ERROR] Failed to initialize model: {exc}")
        sys.exit(1)

    print(f"[INFO] Model path : {args.model}")

    tracker = CentroidTracker(
        max_missing_frames=max_missing_frames,
        max_distance=max_distance,
    )

    source = InputSource(
        args.source,
        path=_resolve_path(args, cfg),
        webcam_index=getattr(cfg, "webcam_index", config.WEBCAM_INDEX),
    )
    fps = FPSCounter()
    window_name = getattr(cfg, "window_name", "DeepX — People Tracking")
    show_fps = getattr(cfg, "show_fps", True)

    print("[INFO] Streaming. Press 'q' or ESC in the window to quit.")
    try:
        while True:
            ok, frame = source.read()
            if not ok or frame is None:
                print("[INFO] Input source exhausted.")
                break

            try:
                boxes, scores, class_ids = model.infer(frame)
            except Exception as exc:
                print(f"[WARN] Inference failed on this frame: {exc}")
                continue

            h, w = frame.shape[:2]

            # Filter to people only BEFORE handing centroids to the tracker.
            # Mixing cars/dogs/people through the same tracker would thrash
            # IDs across class boundaries.
            person_boxes: List[np.ndarray] = []
            centroids: List[Tuple[float, float]] = []
            for box, cid in zip(boxes, class_ids):
                if not _is_person(int(cid), labels):
                    continue
                person_boxes.append(box)
                x1, y1, x2, y2 = box
                # Tracker expects normalized centroids so max_distance is
                # resolution-agnostic.
                cx = ((x1 + x2) * 0.5) / w
                cy = ((y1 + y2) * 0.5) / h
                centroids.append((float(cx), float(cy)))

            # assignments maps index-in-centroids -> track_id.
            assignments = tracker.update(centroids)

            for i, box in enumerate(person_boxes):
                track_id = assignments.get(i)
                if track_id is None:
                    continue
                _draw_person_box(frame, box, track_id)

            # Top-left counter — matches PeopleTrackingAdapter.cpp line 103
            # (result.overlayText = "Total People: N").
            draw_text_lines(
                frame,
                [f"Total People: {len(person_boxes)}"],
                origin=(10, 60),  # Below the FPS counter drawn at y≈30.
            )

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
