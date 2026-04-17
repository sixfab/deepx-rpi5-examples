"""Rich-based keyboard-driven launcher for the DeepX demos.

Run with `python launcher.py` from inside the deepx-demos/ directory.
All paths, sources and thresholds come from ``config.yml``.

Controls:
    ↑ / ↓       move the selection
    Enter       launch the highlighted demo
    q           quit
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text
except ImportError:
    print("Please run: pip install rich")
    sys.exit(1)

from config_loader import get_demo_config
from tui_input import get_keypress, wait_any_key

SCRIPT_DIR = Path(__file__).parent.resolve()
RESOURCES_DIR = SCRIPT_DIR.parent / "resources"
DEMOS_ROOT = str(SCRIPT_DIR)


def _resolve_cfg_path(path: str) -> str:
    """Resolve a config-supplied path to an absolute filesystem path.

    Relative entries are anchored at the launcher directory so the launcher
    behaves identically regardless of the caller's CWD.
    """
    if not path:
        return ""
    p = Path(path)
    if not p.is_absolute():
        p = (SCRIPT_DIR / p).resolve()
    return str(p)

CATEGORY_FOLDERS: Dict[str, str] = {
    "OBJECT DETECTION": "object_detection",
    "CLASSIFICATION": "classification",
    "SEGMENTATION": "segmentation",
    "PPU": "ppu",
    "ASYNC": "async_example",
    "ADVANCED": "advanced",
}

# (category, display_name, config_key, tag_label, tag_color)
DEMOS: List[Tuple[str, str, str, str, str]] = [
    ("OBJECT DETECTION", "scrfd",           "scrfd",           "face",     "green"),
    ("OBJECT DETECTION", "yolov10",         "yolov10",         "detect",   "green"),
    ("OBJECT DETECTION", "yolov11",         "yolov11",         "detect",   "green"),
    ("OBJECT DETECTION", "yolov12",         "yolov12",         "detect",   "green"),
    ("OBJECT DETECTION", "yolov26",         "yolov26",         "detect",   "green"),
    ("OBJECT DETECTION", "yolov26pose",     "yolov26pose",     "pose",     "cyan"),
    ("OBJECT DETECTION", "yolov5",          "yolov5",          "detect",   "green"),
    ("OBJECT DETECTION", "yolov5face",      "yolov5face",      "face",     "green"),
    ("OBJECT DETECTION", "yolov5pose",      "yolov5pose",      "pose",     "cyan"),
    ("OBJECT DETECTION", "yolov7",          "yolov7",          "detect",   "green"),
    ("OBJECT DETECTION", "yolov8",          "yolov8",          "detect",   "green"),
    ("OBJECT DETECTION", "yolov9",          "yolov9",          "detect",   "green"),
    ("OBJECT DETECTION", "yolox",           "yolox",           "detect",   "green"),
    ("CLASSIFICATION",   "yolov26cls",      "yolov26cls",      "classify", "yellow"),
    ("SEGMENTATION",     "deeplabv3",       "deeplabv3",       "semantic", "magenta"),
    ("SEGMENTATION",     "yolov26seg",      "yolov26seg",      "instance", "magenta"),
    ("SEGMENTATION",     "yolov8seg",       "yolov8seg",       "instance", "magenta"),
    ("PPU",              "scrfd_ppu",       "scrfd_ppu",       "face",     "green"),
    ("PPU",              "yolov5_ppu",      "yolov5_ppu",      "detect",   "green"),
    ("PPU",              "yolov5pose_ppu",  "yolov5pose_ppu",  "pose",     "cyan"),
    ("PPU",              "yolov7_ppu",      "yolov7_ppu",      "detect",   "green"),
    ("ASYNC",            "yolov8_async",    "yolov8_async",    "async",    "purple"),
    ("ADVANCED",         "trespassing",          "trespassing",          "zone",     "red"),
    ("ADVANCED",         "people_tracking",      "people_tracking",      "tracking", "cyan"),
    ("ADVANCED",         "smart_traffic",        "smart_traffic",        "counting", "yellow"),
    ("ADVANCED",         "store_queue_analysis", "store_queue_analysis", "queue",    "magenta"),
    ("ADVANCED",         "multi_channel_4",      "multi_channel_4",      "4ch",      "blue"),
    ("ADVANCED",         "hand_landmark",        "hand_landmark",        "landmark", "green"),
]

# Advanced demos rely on structured fields (polygons, lines, channel
# lists) that the other demos don't use. If a user misses them in
# config.yml the demo will still launch but behave strangely — warn at
# startup so they can fix the config before spending time in the UI.
REQUIRED_FIELDS: Dict[str, List[str]] = {
    "trespassing":          ["polygon"],
    "smart_traffic":        ["line"],
    "store_queue_analysis": ["regions"],
    "multi_channel_4":      ["channels"],
}

# Override map for demos whose config key doesn't match `{key}_demo.py`.
# Maps config_key -> (script_path_relative_to_DEMOS_ROOT, demo_key_arg).
# The demo_key_arg is forwarded as --demo-key so the script loads the
# right config block (e.g. one shared script driving multiple variants).
SCRIPT_OVERRIDES: Dict[str, Tuple[str, str]] = {
    "multi_channel_4": ("advanced/multi_channel_demo.py", "multi_channel_4"),
}

console = Console()


def _human_size(path: str) -> str:
    if not os.path.isfile(path):
        return "---"
    n = float(os.path.getsize(path))
    for unit in ("B", "K", "M", "G"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}T"


def _demo_path(config_key: str) -> str:
    """Locate the *_demo.py file for a given config key."""
    override = SCRIPT_OVERRIDES.get(config_key)
    if override is not None:
        candidate = os.path.join(DEMOS_ROOT, override[0])
        if os.path.isfile(candidate):
            return candidate
    for label, folder in CATEGORY_FOLDERS.items():
        candidate = os.path.join(DEMOS_ROOT, folder, f"{config_key}_demo.py")
        if os.path.isfile(candidate):
            return candidate
    return ""


def _build_status_cache() -> Dict[str, bool]:
    """Check each demo's model file once at startup."""
    cache: Dict[str, bool] = {}
    for _cat, _name, key, _tag, _color in DEMOS:
        cfg = get_demo_config(key)
        model = getattr(cfg, "model_path", "") or ""
        resolved = _resolve_cfg_path(model)
        cache[key] = bool(resolved) and os.path.isfile(resolved)
    return cache


def _validate_advanced_configs() -> bool:
    """Print warnings for advanced demos that are missing required fields.

    Returns True iff at least one warning was printed (so the caller can
    pause long enough for the user to actually read them before the Live
    display takes over the terminal).
    """
    printed = False
    for demo_key, required in REQUIRED_FIELDS.items():
        cfg = get_demo_config(demo_key)
        for field in required:
            val = getattr(cfg, field, None)
            if not val:
                console.print(
                    f"[yellow]WARN[/yellow] {demo_key}: "
                    f"'{field}' not set in config.yml — demo may not work correctly."
                )
                printed = True
    return printed


def _source_summary(short: bool = False) -> str:
    cfg = get_demo_config("__summary__")
    source = (getattr(cfg, "input_source", "webcam") or "").lower()
    if source == "video":
        path = getattr(cfg, "video_path", "") or ""
        return f"VIDEO · {os.path.basename(path) if short else path}"
    if source == "image":
        path = getattr(cfg, "image_path", "") or ""
        return f"IMAGE · {os.path.basename(path) if short else path}"
    if source == "rpicam":
        return "RPICAM"
    return f"WEBCAM · index {getattr(cfg, 'webcam_index', 0)}"


def _fmt_item(i: int, selected: int, status_cache: Dict[str, bool]) -> str:
    """Produce the markup for one menu cell, padded to a fixed visible width."""
    _cat, name, key, tag, color = DEMOS[i]
    dot_markup = "[green]●[/green]" if status_cache.get(key) else "[red]○[/red]"

    if i == selected:
        arrow = "[bold purple]▶[/bold purple]"
        idx_markup = f"[bold]{i + 1:>2}[/bold]"
        name_markup = f"[bold white]{name:<16}[/bold white]"
    else:
        arrow = " "
        idx_markup = f"[dim]{i + 1:>2}[/dim]"
        name_markup = f"[dim white]{name:<16}[/dim white]"

    tag_markup = f"[{color}]{tag:<8}[/{color}]"
    body = f" {arrow} {idx_markup}  {name_markup} {tag_markup} {dot_markup} "

    if i == selected:
        return f"[on grey15]{body}[/on grey15]"
    return body


def build_menu(selected: int, status_cache: Dict[str, bool]) -> Group:
    """Return a Rich renderable for the current menu state (flicker-free)."""
    parts: List[Text] = []
    parts.append(Text.from_markup(
        f"[bold purple]SIXFAB[/bold purple] · DeepX Demos   "
        f"[dim]{_source_summary(short=True)}[/dim]"
    ))
    parts.append(Text(""))

    total = len(DEMOS)
    half = (total + 1) // 2
    for row in range(half):
        left = _fmt_item(row, selected, status_cache)
        right_i = row + half
        right = _fmt_item(right_i, selected, status_cache) if right_i < total else ""
        parts.append(Text.from_markup(left + "  " + right))

    sel_name = DEMOS[selected][1]
    sel_key = DEMOS[selected][2]
    cfg = get_demo_config(sel_key)
    model_path = getattr(cfg, "model_path", "") or "(no model_path)"
    size = _human_size(_resolve_cfg_path(model_path))
    parts.append(Text(""))
    parts.append(Text.from_markup(
        f"[dim]selected:[/dim] [bold]{sel_name}[/bold]  "
        f"[dim]{model_path}[/dim]  [dim]({size})[/dim]"
    ))
    parts.append(Text.from_markup("[dim]↑↓ navigate   Enter launch   q quit[/dim]"))
    return Group(*parts)


def _build_cmd(demo_path: str, cfg) -> List[str]:
    cmd = [sys.executable, demo_path]
    source = getattr(cfg, "input_source", "webcam")
    cmd += ["--source", source]
    if source == "video" and getattr(cfg, "video_path", None):
        cmd += ["--path", _resolve_cfg_path(cfg.video_path)]
    elif source == "image" and getattr(cfg, "image_path", None):
        cmd += ["--path", _resolve_cfg_path(cfg.image_path)]

    if getattr(cfg, "model_path", None):
        cmd += ["--model", _resolve_cfg_path(cfg.model_path)]
    if getattr(cfg, "label_path", None):
        cmd += ["--labels", _resolve_cfg_path(cfg.label_path)]
    if getattr(cfg, "confidence_threshold", None) is not None:
        cmd += ["--conf", str(cfg.confidence_threshold)]
    if getattr(cfg, "iou_threshold", None) is not None:
        cmd += ["--iou", str(cfg.iou_threshold)]
    return cmd


def launch_demo(index: int, status_cache: Dict[str, bool]) -> None:
    _cat, name, key, _tag, _color = DEMOS[index]
    cfg = get_demo_config(key)
    model_path = getattr(cfg, "model_path", "") or ""
    source = (getattr(cfg, "input_source", "webcam") or "webcam").upper()
    size = _human_size(_resolve_cfg_path(model_path))

    console.clear()

    body = Text()
    body.append(f"Launching: ", style="dim")
    body.append(f"{name}\n", style="bold white")
    body.append(f"Model    : ", style="dim")
    body.append(f"{os.path.basename(model_path) or '(none)'}  {size}\n", style="white")
    body.append(f"Source   : ", style="dim")
    body.append(source, style="white")
    console.print(Panel(body, border_style="purple", title="[bold]DeepX[/bold]"))

    if not status_cache.get(key):
        console.print(
            f"[red]WARNING:[/red] Model file not found: [bold]{model_path}[/bold]"
        )
        console.print("[dim]Please update config.yml and try again.[/dim]")
        console.print()
        console.print("[dim]Press any key to return to the menu...[/dim]")
        wait_any_key()
        return

    demo_path = _demo_path(key)
    if not demo_path:
        console.print(f"[red]ERROR:[/red] demo file not found for '{key}'")
        console.print("[dim]Press any key to return to the menu...[/dim]")
        wait_any_key()
        return

    cmd = _build_cmd(demo_path, cfg)
    # When this entry maps to a shared script, the script needs --demo-key
    # to know which config block to load.
    override = SCRIPT_OVERRIDES.get(key)
    if override is not None:
        cmd += ["--demo-key", override[1]]
    console.print()
    try:
        subprocess.run(cmd, cwd=DEMOS_ROOT)
    except KeyboardInterrupt:
        console.print("\n[dim][stopped][/dim]")

    console.print()
    console.print("[dim]Demo finished. Press any key to return...[/dim]")
    wait_any_key()


def main() -> None:
    # Validate before building the status cache so early warnings surface
    # above the menu. If anything was printed we pause briefly — the
    # upcoming Live panel will otherwise redraw right over the warnings.
    if _validate_advanced_configs():
        time.sleep(2.0)

    status_cache = _build_status_cache()
    selected = 0
    total = len(DEMOS)

    while True:
        quitting = False
        with Live(
            build_menu(selected, status_cache),
            console=console,
            auto_refresh=False,
            transient=True,
            screen=False,
        ) as live:
            while True:
                key = get_keypress()
                if key == 'up':
                    selected = (selected - 1) % total
                elif key == 'down':
                    selected = (selected + 1) % total
                elif key == 'enter':
                    break
                elif key == 'q':
                    quitting = True
                    break
                else:
                    continue
                live.update(build_menu(selected, status_cache), refresh=True)

        if quitting:
            return
        launch_demo(selected, status_cache)


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        console.clear()
