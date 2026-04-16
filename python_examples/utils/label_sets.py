"""Built-in label sets used as fallbacks when no LABEL_PATH file exists.

Only the labels actually consumed by the demos are stored here; this keeps
the file small and readable. ImageNet's full 1000-class list is loaded
lazily because it is large.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List

# COCO 80-class detection labels (YOLO family default).
COCO80: List[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

# DOTA v1 oriented-bounding-box dataset (used by yolov26-OBB).
DOTAV1: List[str] = [
    "plane", "ship", "storage-tank", "baseball-diamond", "tennis-court",
    "basketball-court", "ground-track-field", "harbor", "bridge",
    "large-vehicle", "small-vehicle", "helicopter", "roundabout",
    "soccer-ball-field", "swimming-pool",
]


@lru_cache(maxsize=1)
def imagenet1000() -> List[str]:
    """Return ImageNet-1k labels.

    Loaded from the SDK example pack if available so we do not duplicate
    the full 1000-line list inside this repository. Falls back to numeric
    indices when no source is found, which still lets the demo run.
    """
    candidates = [
        # Shipped with the upstream DeepX SDK examples.
        os.path.join(
            os.path.dirname(__file__), "..", "..", "example_pythons", "utils", "labels.py"
        ),
    ]
    for path in candidates:
        path = os.path.abspath(path)
        if not os.path.exists(path):
            continue
        try:
            namespace: dict = {}
            with open(path, "r", encoding="utf-8") as fh:
                exec(fh.read(), namespace)  # noqa: S102 — trusted local file
            if "IMAGENET_1000" in namespace:
                return list(namespace["IMAGENET_1000"])
        except Exception:
            continue
    return [f"class_{i}" for i in range(1000)]


# Cityscapes 19-class palette used by deeplabv3 semantic-seg models.
CITYSCAPES_PALETTE = [
    (128, 64, 128),  (244, 35, 232),  (70, 70, 70),    (102, 102, 156),
    (190, 153, 153), (153, 153, 153), (250, 170, 30),  (220, 220, 0),
    (107, 142, 35),  (152, 251, 152), (70, 130, 180),  (220, 20, 60),
    (255, 0, 0),     (0, 0, 142),     (0, 0, 70),      (0, 60, 100),
    (0, 80, 100),    (0, 0, 230),     (119, 11, 32),
]
