"""
config_loader.py — Loads and merges config.yml for DeepX demos.

Usage:
    from config_loader import get_demo_config
    cfg = get_demo_config("yolov8")
    # cfg.model_path, cfg.video_path, cfg.confidence_threshold, etc.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import Any, Dict

import yaml

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yml")


def _load_raw() -> Dict[str, Any]:
    if not os.path.isfile(CONFIG_PATH):
        print(
            "ERROR: config.yml not found. Please create it next to the launcher. "
            "See README for instructions."
        )
        sys.exit(1)
    with open(CONFIG_PATH, "r") as f:
        data = yaml.safe_load(f) or {}
    return data


def _normalize_structured_fields(base: Dict[str, Any]) -> None:
    """Convert raw YAML shapes into the tuple/list forms demos expect.

    Advanced demos (trespassing, smart_traffic, store_queue_analysis, …)
    ship structured geometry in config.yml. YAML decodes nested sequences
    as plain lists; we coerce them into (x, y) tuples here so demo code
    can treat coordinates as hashable, immutable pairs.
    """
    # polygon: [[x,y], ...] -> [(x,y), ...]
    polygon = base.get("polygon")
    if polygon is not None:
        base["polygon"] = [(float(p[0]), float(p[1])) for p in polygon]

    # line: [x1, y1, x2, y2] -> ((x1, y1), (x2, y2))
    line = base.get("line")
    if line is not None and len(line) == 4:
        base["line"] = (
            (float(line[0]), float(line[1])),
            (float(line[2]), float(line[3])),
        )

    # regions: [[[x,y], ...], ...] -> [[(x,y), ...], ...]
    regions = base.get("regions")
    if regions is not None:
        base["regions"] = [
            [(float(p[0]), float(p[1])) for p in region] for region in regions
        ]

    # channels: list of {source, path} dicts — pass through as plain dicts
    # but guarantee the expected keys exist so demo code can destructure
    # without defensive .get() calls.
    channels = base.get("channels")
    if channels is not None:
        base["channels"] = [
            {"source": ch.get("source", "webcam"), "path": ch.get("path")}
            for ch in channels
        ]

    # vehicle_classes: list of strings (YAML already gives us this shape,
    # but coerce defensively in case someone writes a single string).
    vehicle_classes = base.get("vehicle_classes")
    if vehicle_classes is not None and not isinstance(vehicle_classes, list):
        base["vehicle_classes"] = [str(vehicle_classes)]

    # wait_thresholds: {green: int, yellow: int}
    wait_thresholds = base.get("wait_thresholds")
    if wait_thresholds is not None:
        base["wait_thresholds"] = {
            "green": int(wait_thresholds.get("green", 7)),
            "yellow": int(wait_thresholds.get("yellow", 15)),
        }

    # Model decode params (mirror the JSON-driven C++ demos). These get
    # forwarded to the PPU postprocess in advanced/ — they're optional for
    # everything else, so we only normalize the types here when present.
    if "ultralytics" in base:
        base["ultralytics"] = bool(base["ultralytics"])
    if "input_width" in base:
        base["input_width"] = int(base["input_width"])
    if "input_height" in base:
        base["input_height"] = int(base["input_height"])
    if "decoding_method" in base:
        base["decoding_method"] = str(base["decoding_method"])
    if "box_format" in base:
        base["box_format"] = str(base["box_format"])
    if "last_activation" in base:
        base["last_activation"] = str(base["last_activation"])

    # anchors: list of {stride, widths, heights} dicts. We coerce ints/floats
    # so demo code can hand the lists straight to numpy without per-row casts.
    anchors = base.get("anchors")
    if anchors is not None:
        base["anchors"] = [
            {
                "stride": int(a["stride"]),
                "widths": [float(x) for x in a["widths"]],
                "heights": [float(x) for x in a["heights"]],
            }
            for a in anchors
        ]


def get_demo_config(demo_name: str) -> SimpleNamespace:
    """Return merged config for `demo_name` as a SimpleNamespace.

    Global values form the base. Values under ``demos.<demo_name>`` override
    the globals. Missing demo entries silently fall back to globals.

    Structured fields used by the advanced demos are normalized into the
    forms demo code expects:
        - polygon          -> list[tuple[float, float]]
        - line             -> tuple[tuple[float, float], tuple[float, float]]
        - regions          -> list[list[tuple[float, float]]]
        - channels         -> list[dict]  (keys: 'source', 'path')
        - vehicle_classes  -> list[str]
        - wait_thresholds  -> dict        (keys: 'green', 'yellow')
        - ultralytics      -> bool        (anchor-free yolov8 head if True)
        - input_width      -> int
        - input_height     -> int
        - decoding_method  -> str
        - box_format       -> str
        - last_activation  -> str
        - anchors          -> list[dict]  (keys: 'stride', 'widths', 'heights')
    Missing fields are left unset; demos should use ``getattr(cfg, name, default)``.
    """
    data = _load_raw()
    base: Dict[str, Any] = dict(data.get("global") or {})
    demo_overrides: Dict[str, Any] = dict((data.get("demos") or {}).get(demo_name) or {})
    base.update(demo_overrides)
    base["demo_name"] = demo_name
    _normalize_structured_fields(base)
    return SimpleNamespace(**base)


def validate_paths(cfg: SimpleNamespace) -> bool:
    """Check that files referenced by `cfg` exist on disk.

    Prints a WARNING for each missing file. Returns True when every required
    path exists, False otherwise. Does not raise or exit.
    """
    ok = True

    model_path = getattr(cfg, "model_path", None)
    if model_path and not os.path.isfile(model_path):
        print(f"WARNING: Model file not found: {model_path}")
        ok = False

    source = getattr(cfg, "input_source", "webcam")
    if source == "video":
        video_path = getattr(cfg, "video_path", None)
        if video_path and not os.path.isfile(video_path):
            print(f"WARNING: Video file not found: {video_path}")
            ok = False
    elif source == "image":
        image_path = getattr(cfg, "image_path", None)
        if image_path and not os.path.isfile(image_path):
            print(f"WARNING: Image file not found: {image_path}")
            ok = False

    label_path = getattr(cfg, "label_path", None)
    if label_path and not os.path.isfile(label_path):
        print(f"WARNING: Label file not found: {label_path}")
        ok = False

    return ok
