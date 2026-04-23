"""Smart Traffic Counter Demo — DeepX SDK
----------------------------------------
Counts vehicles crossing a user-defined line.
Each vehicle gets a track ID. Crossing is detected via segment intersection
between the vehicle's previous and current centroid.

Based on: SmartTrafficAdapter.cpp

Note: yolov8l-ppu.dxnn is a PPU-compiled model that emits a packed
(1, N, 32) byte buffer rather than the classic (1, 84, N) float tensor.
Decoding goes through ``utils.ppu`` (anchor-free yolov8 head).

Config (config.yml → demos.smart_traffic):
  model_path         : Path to YOLO .dxnn model (YOLOv8-family expected)
  line               : [x1, y1, x2, y2] normalized counting line
  vehicle_classes    : List of class names to count
  max_missing_frames : Frames before a track is pruned
  max_match_distance : Normalized distance threshold for matching
  confidence_threshold
  ultralytics        : True for the yolov8 PPU head used here

Usage:
  python advanced/smart_traffic_demo.py
  python advanced/smart_traffic_demo.py --source video --path traffic.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Sequence, Set, Tuple

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
from utils.zone_utils import check_line_crossing, draw_line_overlay


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


# Fallback used when config.yml is missing the 'line' field. A horizontal
# line across the vertical middle is the most common traffic setup.
_DEFAULT_LINE = ((0.0, 0.5), (1.0, 0.5))
_DEFAULT_VEHICLE_CLASSES = ("car", "truck", "bus", "motorcycle")


def parse_args(cfg) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smart Traffic Counter Demo — DeepX SDK")
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


def _build_vehicle_id_set(
    vehicle_classes: Sequence[str], labels: Sequence[str]
) -> Set[int]:
    """Translate vehicle-class names into the integer ids we'll match against.

    Mirrors SmartTrafficAdapter.cpp lines 32-39: the C++ version does a
    string equality check per detection per frame; here we resolve the
    strings to ids once, then do cheap int-set lookups in the hot loop.
    """
    wanted = {c.lower() for c in vehicle_classes}
    return {i for i, lbl in enumerate(labels) if lbl.lower() in wanted}


def _draw_vehicle_box(
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
    cfg = get_demo_config("smart_traffic")
    args = parse_args(cfg)

    line = getattr(cfg, "line", None)
    if not line or len(line) != 2:
        print("[WARN] No valid counting line in config.yml — using horizontal centre line.")
        line = _DEFAULT_LINE
    line_start, line_end = line

    vehicle_classes = getattr(cfg, "vehicle_classes", None) or []
    if not vehicle_classes:
        print(
            f"[WARN] No vehicle_classes configured — defaulting to {list(_DEFAULT_VEHICLE_CLASSES)}."
        )
        vehicle_classes = list(_DEFAULT_VEHICLE_CLASSES)

    max_missing_frames = int(getattr(cfg, "max_missing_frames", 15))
    max_match_distance = float(getattr(cfg, "max_match_distance", 0.1))

    labels = sdk.load_labels(args.labels, COCO80)
    vehicle_ids = _build_vehicle_id_set(vehicle_classes, labels)
    if not vehicle_ids:
        print(
            f"[WARN] None of vehicle_classes={vehicle_classes} appear in the "
            f"labels file — the counter will always read zero."
        )

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
        max_distance=max_match_distance,
    )

    # State carried across frames. The tracker handles ID persistence; we
    # layer crossing-state on top of it. Both dicts are keyed by track_id
    # and pruned together with the tracker's own active set.
    prev_centroids: Dict[int, Tuple[float, float]] = {}
    crossed: Set[int] = set()
    total_count = 0

    source = InputSource(
        args.source,
        path=_resolve_path(args, cfg),
        webcam_index=getattr(cfg, "webcam_index", config.WEBCAM_INDEX),
    )
    fps = FPSCounter()
    window_name = getattr(cfg, "window_name", "DeepX — Smart Traffic")
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

            # Filter to the vehicle classes of interest before tracking so
            # we don't cross cars and people through the same ID pool.
            vehicle_boxes: List[np.ndarray] = []
            vehicle_class_ids: List[int] = []
            centroids: List[Tuple[float, float]] = []
            for box, cid in zip(boxes, class_ids):
                cid_int = int(cid)
                if cid_int not in vehicle_ids:
                    continue
                vehicle_boxes.append(box)
                vehicle_class_ids.append(cid_int)
                x1, y1, x2, y2 = box
                cx = ((x1 + x2) * 0.5) / w
                cy = ((y1 + y2) * 0.5) / h
                centroids.append((float(cx), float(cy)))

            assignments = tracker.update(centroids)

            # Step through each tracked vehicle and decide whether this
            # frame's motion crossed the counting line. Every frame after
            # the first one has a valid prev centroid; the very first frame
            # a track appears we simply record its position.
            for det_idx, track_id in assignments.items():
                curr = centroids[det_idx]
                prev = prev_centroids.get(track_id)

                if prev is not None and track_id not in crossed:
                    if check_line_crossing(prev, curr, line_start, line_end):
                        total_count += 1
                        crossed.add(track_id)

                # Update prev AFTER the crossing check — otherwise prev
                # and curr would always be identical and nothing would
                # ever cross.
                prev_centroids[track_id] = curr

            # Prune state for tracks the CentroidTracker has already dropped,
            # so these dicts don't grow without bound on long videos.
            active_ids = set(tracker.get_active_tracks().keys())
            prev_centroids = {tid: c for tid, c in prev_centroids.items() if tid in active_ids}
            crossed &= active_ids

            # Draw order: line under boxes, boxes with per-track colour,
            # counter overlay last so it sits on top of everything.
            draw_line_overlay(frame, line_start, line_end, color=(0, 255, 0), thickness=2)

            for det_idx, box in enumerate(vehicle_boxes):
                track_id = assignments.get(det_idx)
                if track_id is None:
                    continue
                cid = vehicle_class_ids[det_idx]
                name = labels[cid] if 0 <= cid < len(labels) else "vehicle"
                suffix = " ✓" if track_id in crossed else ""
                # "✓" is cosmetic; OpenCV Hershey can't render it, so strip
                # to ASCII. The trailing flag is useful when debugging.
                label = f"{name} #{track_id}{'' if not suffix else ' (counted)'}"
                _draw_vehicle_box(frame, box, label, color_for(track_id))

            draw_text_lines(
                frame,
                [f"Count: {total_count}"],
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
