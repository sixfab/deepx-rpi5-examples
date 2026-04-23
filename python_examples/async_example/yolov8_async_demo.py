"""YOLOv8 Async Pipeline Demo — DeepX SDK
--------------------------------------------
Description: Same YOLOv8 model as the sync demo, but driven by a 3-stage
             pipeline that overlaps preprocess → inference → postprocess
             across worker threads. Each stage runs on its own thread and
             passes work via bounded queues. This dramatically increases
             throughput on long videos but adds a few frames of end-to-end
             latency, so it is the right tool for "process this batch as
             fast as possible" workloads and the wrong tool for low-latency
             interactive UIs.

             Trade-off summary:
                 sync mode    -> low latency,  low throughput
                 async mode   -> higher latency, higher throughput

Input      : webcam / rpicam / video file / image file
Output     : OpenCV window with bounding boxes; FPS counted at the renderer.

Usage:
    python yolov8_async_demo.py                              # uses config.py
    python yolov8_async_demo.py --source video --path drive.mp4
    python yolov8_async_demo.py --model yolov8.dxnn --queue 4
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
from typing import Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from utils import sdk
from utils.fps_counter import FPSCounter
from utils.input_source import InputSource
from utils.label_sets import COCO80
from utils.visualizer import draw_detections

DEFAULT_QUEUE_SIZE = 4   # How many frames may be in flight at once.
SENTINEL = object()      # Marker used to tell workers it is time to quit.


class YOLOv8Async:
    """YOLOv8 model wrapper that exposes both sync and async run paths.

    Most demos in this repo just call `engine.run([tensor])`. This one
    instead calls `engine.run_async(...)` to get a request id and later
    `engine.wait(req_id)` to fetch the outputs. That split is what lets
    a separate thread overlap host-side work with NPU work.
    """

    def __init__(self, model_path: str, conf_threshold: float,
                 iou_threshold: float) -> None:
        self.engine, self.input_h, self.input_w = sdk.load_engine(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    # The geometry context (gain, pad, src shape) lives on each work item
    # rather than on `self`, because in async mode multiple frames may be
    # in flight at once and they would otherwise stomp on each other.

    def preprocess(self, frame_bgr: np.ndarray):
        src_shape = frame_bgr.shape[:2]
        rgb = sdk.bgr_to_rgb(frame_bgr)
        padded, gain, pad = sdk.letterbox(rgb, (self.input_h, self.input_w))
        return padded, {"gain": gain, "pad": pad, "src_shape": src_shape}

    def postprocess(self, output_tensors, geom):
        outputs = np.transpose(np.squeeze(output_tensors[0]))
        cls_scores = outputs[:, 4:]
        scores = np.max(cls_scores, axis=1)
        class_ids = np.argmax(cls_scores, axis=1)
        boxes = sdk.cxcywh_to_xyxy(outputs[:, :4])

        keep = sdk.nms(boxes, scores, self.conf_threshold, self.iou_threshold)
        if keep.size == 0:
            empty = np.empty((0,), dtype=np.float32)
            return np.empty((0, 4), dtype=np.float32), empty, empty.astype(np.int64)

        boxes = sdk.unletterbox_boxes(
            boxes[keep], geom["gain"], geom["pad"], geom["src_shape"]
        )
        return boxes, scores[keep], class_ids[keep]


def _drain_queue(q: queue.Queue) -> None:
    """Pop everything from a queue without blocking. Used at shutdown."""
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        pass


def run_pipeline(model: YOLOv8Async, source: InputSource,
                 labels, queue_size: int, show: bool) -> None:
    """Three-stage pipeline: preprocess → wait → postprocess+draw.

    Each `queue.Queue(maxsize=queue_size)` acts as backpressure: when a
    downstream stage is slow, upstream stages naturally block on `put()`
    instead of growing memory without bound.
    """

    pre_q: "queue.Queue[Tuple[np.ndarray, np.ndarray, dict]]" = queue.Queue(
        maxsize=queue_size
    )
    wait_q: "queue.Queue[Tuple[np.ndarray, object, dict]]" = queue.Queue(
        maxsize=queue_size
    )

    stop = threading.Event()
    fps = FPSCounter()

    def producer():
        """Read frames + run preprocess, then submit to the NPU."""
        try:
            while not stop.is_set():
                ok, frame = source.read()
                if not ok or frame is None:
                    break
                input_tensor, geom = model.preprocess(frame)
                # Submit to the NPU before blocking on the queue so that
                # the device starts working as early as possible.
                req_id = model.engine.run_async([input_tensor])
                pre_q.put((frame, input_tensor, geom, req_id))
        except Exception as exc:
            print(f"[ERROR] Producer thread: {exc}")
            stop.set()
        finally:
            pre_q.put(SENTINEL)

    def waiter():
        """Block on each request id and forward the raw outputs."""
        try:
            while True:
                item = pre_q.get()
                if item is SENTINEL or stop.is_set():
                    break
                frame, input_tensor, geom, req_id = item
                # input_tensor is kept alive in the tuple so its memory
                # cannot be reused by NumPy while the NPU is still reading.
                outputs = model.engine.wait(req_id)
                wait_q.put((frame, outputs, geom))
        except Exception as exc:
            print(f"[ERROR] Waiter thread: {exc}")
            stop.set()
        finally:
            wait_q.put(SENTINEL)

    # The consumer (postprocess + draw + display) runs on the main thread
    # because OpenCV's HighGUI must be called from the thread that created
    # the window. Workers above only touch numpy and SDK calls.
    print("[INFO] Async pipeline started. Press 'q' or ESC to quit.")
    threads = [
        threading.Thread(target=producer, daemon=True),
        threading.Thread(target=waiter, daemon=True),
    ]
    for t in threads:
        t.start()

    try:
        while True:
            item = wait_q.get()
            if item is SENTINEL:
                print("[INFO] Input source exhausted.")
                break
            frame, outputs, geom = item

            try:
                boxes, scores, class_ids = model.postprocess(outputs, geom)
            except Exception as exc:
                print(f"[WARN] Postprocess failed for one frame: {exc}")
                continue

            draw_detections(frame, boxes, scores, class_ids, labels, config)

            fps.update()
            if config.SHOW_FPS:
                fps.draw(frame)

            if show:
                cv2.imshow(config.WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    print("[INFO] Quit requested by user.")
                    break

    except KeyboardInterrupt:
        print("[INFO] Interrupted (Ctrl+C).")
    finally:
        stop.set()
        # Wake any worker blocked on a full queue.
        _drain_queue(pre_q)
        _drain_queue(wait_q)
        for t in threads:
            t.join(timeout=2.0)
        source.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLOv8 Async Demo — DeepX SDK")
    p.add_argument("--source", type=str, default=config.INPUT_SOURCE,
                   choices=list(InputSource.SUPPORTED))
    p.add_argument("--path", type=str, default=None)
    p.add_argument("--model", type=str, default=config.MODEL_PATH)
    p.add_argument("--labels", type=str, default=config.LABEL_PATH)
    p.add_argument("--conf", type=float, default=config.CONFIDENCE_THRESHOLD)
    p.add_argument("--iou", type=float, default=config.IOU_THRESHOLD)
    p.add_argument("--queue", type=int, default=DEFAULT_QUEUE_SIZE,
                   help="Max in-flight frames between pipeline stages.")
    p.add_argument("--no-display", dest="display", action="store_false")
    p.set_defaults(display=True)
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
    labels = sdk.load_labels(args.labels, COCO80)
    model = YOLOv8Async(args.model, args.conf, args.iou)
    source = InputSource(args.source, path=_resolve_path(args),
                         webcam_index=config.WEBCAM_INDEX)

    run_pipeline(model, source, labels, queue_size=max(1, args.queue),
                 show=args.display)


if __name__ == "__main__":
    main()
