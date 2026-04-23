"""Store Queue Analysis Demo — DeepX SDK
---------------------------------------
Tracks people standing inside one or more queue zones. Each person's
bounding box is colour-coded by how long they have been in the queue:

    Green  — waiting < 7 seconds
    Yellow — waiting 7-15 seconds
    Red    — waiting > 15 seconds

Based on: StoreQueueAnalysisAdapter.cpp

Note: yolov8l-ppu.dxnn is a PPU-compiled model that emits a packed
(1, N, 32) byte buffer rather than the classic (1, 84, N) float tensor.
Decoding goes through ``utils.ppu`` (anchor-free yolov8 head).

Config (config.yml → demos.store_queue_analysis):
  model_path          : Path to YOLO .dxnn model
  regions             : List of polygons (list of [x,y] points) in normalized coords
  wait_thresholds     : {green: 7, yellow: 15}   seconds
  max_missing_frames  : Frames before a track is pruned
  max_distance        : Normalized centroid-matching threshold
  confidence_threshold
  ultralytics         : True for the yolov8 PPU head used here

Usage:
  python advanced/store_queue_analysis_demo.py
  python advanced/store_queue_analysis_demo.py --source video --path store.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Sequence, Tuple

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
from utils.visualizer import draw_text_lines
from utils.zone_utils import draw_multi_polygon_overlay, point_in_polygon


class PpuDetector:
    """PPU-decoded YOLO detector. Branches on ``ultralytics`` for the head."""

    def __init__(
        self,
        model_path: str,
        labels: Sequence[str],
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
        outputs = self.engine.run([self.preprocess(frame_bgr)])
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


_PERSON_CLASS_ID = 0

# Full-frame fallback when config has no valid regions. Makes the miswiring
# visually obvious (every person will be coloured by wait time).
_DEFAULT_REGIONS = [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]]


def parse_args(cfg) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Store Queue Analysis Demo — DeepX SDK")
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
    if class_id == _PERSON_CLASS_ID:
        return True
    if 0 <= class_id < len(labels):
        return labels[class_id].lower() == "person"
    return False


def _inside_any_region(
    point: Tuple[float, float],
    regions: Sequence[Sequence[Tuple[float, float]]],
) -> bool:
    """True if `point` is inside at least one of the configured polygons.

    Port of StoreQueueAnalysisAdapter.cpp lines 83-93: a person counts as
    queueing if they are inside *any* region, not only the nearest one.
    """
    for polygon in regions:
        if point_in_polygon(point, polygon):
            return True
    return False


def _wait_color(
    duration_s: float, green_t: float, yellow_t: float
) -> tuple:
    """Traffic-light colour mapping for wait-time visualisation.

    Port of StoreQueueAnalysisAdapter.cpp lines 105-116. Returned values
    are BGR tuples, ready to hand to cv2.rectangle.
    """
    if duration_s < green_t:
        return (0, 255, 0)      # Green   — still fresh
    if duration_s < yellow_t:
        return (0, 255, 255)    # Yellow  — getting long
    return (0, 0, 255)          # Red     — too long


def _draw_person_box(
    frame: np.ndarray,
    box: np.ndarray,
    label: str,
    color: tuple,
) -> None:
    x1, y1, x2, y2 = box.astype(int)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    label_y = y1 - 6 if y1 - th - 6 > 0 else y1 + th + 6
    cv2.rectangle(
        frame,
        (x1, label_y - th - 4),
        (x1 + tw + 4, label_y + 4),
        color,
        cv2.FILLED,
    )
    cv2.putText(
        frame, label, (x1 + 2, label_y), font, scale, (0, 0, 0), thickness, cv2.LINE_AA
    )


def main() -> None:
    cfg = get_demo_config("store_queue_analysis")
    args = parse_args(cfg)

    regions = getattr(cfg, "regions", None) or []
    regions = [r for r in regions if len(r) >= 3]
    if not regions:
        print(
            "[WARN] No valid queue regions in config.yml "
            "(need >=3 points per region) — defaulting to full frame."
        )
        regions = _DEFAULT_REGIONS

    wait_thresholds = getattr(cfg, "wait_thresholds", None) or {}
    green_t = float(wait_thresholds.get("green", 7))
    yellow_t = float(wait_thresholds.get("yellow", 15))

    max_missing_frames = int(getattr(cfg, "max_missing_frames", 10))
    max_distance = float(getattr(cfg, "max_distance", 0.1))

    labels = sdk.load_labels(args.labels, COCO80)

    ultralytics = bool(getattr(cfg, "ultralytics", True))
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

    # Per-track queue state. 'enter_time' uses time.monotonic() so the
    # wait duration is unaffected by wall-clock changes (NTP jumps, DST).
    queue_state: Dict[int, Dict] = {}

    source = InputSource(
        args.source,
        path=_resolve_path(args, cfg),
        webcam_index=getattr(cfg, "webcam_index", config.WEBCAM_INDEX),
    )
    fps = FPSCounter()
    window_name = getattr(cfg, "window_name", "DeepX — Store Queue Analysis")
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
            now = time.monotonic()

            # Collect person detections and their normalized centroids.
            person_boxes: List[np.ndarray] = []
            centroids: List[Tuple[float, float]] = []
            for box, cid in zip(boxes, class_ids):
                if not _is_person(int(cid), labels):
                    continue
                person_boxes.append(box)
                x1, y1, x2, y2 = box
                cx = ((x1 + x2) * 0.5) / w
                cy = ((y1 + y2) * 0.5) / h
                centroids.append((float(cx), float(cy)))

            assignments = tracker.update(centroids)

            # Update queue membership per track. A track's state flips only
            # on the transition in/out of a region — while it stays inside,
            # we keep the original enter_time so the clock keeps running.
            queue_count = 0
            draw_list: List[Tuple[np.ndarray, str, tuple]] = []

            for det_idx, box in enumerate(person_boxes):
                track_id = assignments.get(det_idx)
                if track_id is None:
                    continue

                state = queue_state.setdefault(
                    track_id, {"in_queue": False, "enter_time": None}
                )
                inside = _inside_any_region(centroids[det_idx], regions)

                if inside and not state["in_queue"]:
                    # Fresh entrant — start their wait-time clock.
                    state["in_queue"] = True
                    state["enter_time"] = now
                elif not inside and state["in_queue"]:
                    # They've left the queue; reset so a future re-entry
                    # starts counting again from zero.
                    state["in_queue"] = False
                    state["enter_time"] = None

                if state["in_queue"] and state["enter_time"] is not None:
                    duration = now - state["enter_time"]
                    color = _wait_color(duration, green_t, yellow_t)
                    label = f"In Queue {track_id}: {int(duration)}s"
                    queue_count += 1
                else:
                    color = (255, 255, 255)
                    label = f"Person {track_id}"

                draw_list.append((box, label, color))

            # Prune queue_state entries whose tracks have been pruned by
            # the CentroidTracker.
            active_ids = set(tracker.get_active_tracks().keys())
            queue_state = {tid: s for tid, s in queue_state.items() if tid in active_ids}

            # Draw order: regions (translucent) → people → overlay text.
            draw_multi_polygon_overlay(frame, regions, color=(255, 200, 0), alpha=0.2)

            for box, label, color in draw_list:
                _draw_person_box(frame, box, label, color)

            draw_text_lines(
                frame,
                [f"Queue Count: {queue_count}"],
                origin=(10, 60),
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
