# Python Examples

A full collection of 28 inference demos for the DEEPX NPU, launched from a keyboard-driven TUI menu. Covers everything from basic object detection to advanced zone analytics and multi-channel inference.

## Table of Contents
- [Quick Start](#quick-start)
- [The Launcher](#the-launcher)
- [Available Demos](#available-demos)
- [Configuration](#configuration)
- [Running Demos Directly](#running-demos-directly)
- [Keyboard Controls](#keyboard-controls)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

Run `auto-install.sh` from the repo root first (see [install-raspberry-pi5.md](install-raspberry-pi5.md)), then:

```bash
bash python_examples/start.sh
```

`start.sh` activates `/opt/sixfab-dx/venv`, installs Python dependencies, downloads models and videos if they are missing, then opens the demo launcher.

---

## The Launcher

`python_examples/launcher.py` is a Rich-based TUI that lists all 28 demos.

- Navigate with `↑` / `↓` arrow keys
- Press `Enter` to launch the selected demo
- Press `q` to quit
- A green `●` means the model file is present on disk
- A red `○` means the model file is missing — update `model_path` for that demo in `config.yml`

Demos launch immediately using the settings from `config.yml`. There are no interactive prompts during launch.

---

## Available Demos

### Object Detection (13 demos)

| # | Demo key | Description |
|---|----------|-------------|
| 1 | `scrfd` | Face detection with SCRFD |
| 2 | `yolov10` | General object detection (YOLOv10) |
| 3 | `yolov11` | General object detection (YOLOv11) |
| 4 | `yolov12` | General object detection (YOLOv12) |
| 5 | `yolov26` | General object detection (YOLOv26) |
| 6 | `yolov26pose` | Human pose estimation — skeleton keypoints |
| 7 | `yolov5` | General object detection (YOLOv5) |
| 8 | `yolov5face` | Face detection with YOLOv5-face |
| 9 | `yolov5pose` | Human pose estimation — skeleton keypoints |
| 10 | `yolov7` | General object detection (YOLOv7) |
| 11 | `yolov8` | General object detection (YOLOv8) |
| 12 | `yolov9` | General object detection (YOLOv9) |
| 13 | `yolox` | General object detection (YOLOX) |

### Classification (1 demo)

| # | Demo key | Description |
|---|----------|-------------|
| 14 | `yolov26cls` | Image classification with YOLOv26-cls |

### Segmentation (3 demos)

| # | Demo key | Description |
|---|----------|-------------|
| 15 | `deeplabv3` | Semantic segmentation with DeepLabV3+ |
| 16 | `yolov26seg` | Instance segmentation with YOLOv26-seg |
| 17 | `yolov8seg` | Instance segmentation with YOLOv8-seg |

### PPU — Hardware-Preprocessed Variants (4 demos)

PPU demos offload image preprocessing to dedicated hardware on the NPU module, freeing CPU resources and increasing throughput.

| # | Demo key | Description |
|---|----------|-------------|
| 18 | `scrfd_ppu` | SCRFD face detection (PPU) |
| 19 | `yolov5_ppu` | YOLOv5 detection (PPU) |
| 20 | `yolov5pose_ppu` | YOLOv5 pose estimation (PPU) |
| 21 | `yolov7_ppu` | YOLOv7 detection (PPU) |

### Async Inference (1 demo)

| # | Demo key | Description |
|---|----------|-------------|
| 22 | `yolov8_async` | Async inference pipeline — demonstrates non-blocking RunAsync / Wait pattern |

### Advanced Analytics (6 demos)

| # | Demo key | Description |
|---|----------|-------------|
| 23 | `trespassing` | Forbidden-zone intrusion alert with configurable polygon boundary |
| 24 | `people_tracking` | Person tracking with persistent IDs across frames |
| 25 | `smart_traffic` | Vehicle line-crossing counter with configurable counting line |
| 26 | `store_queue_analysis` | Queue wait-time analyzer with color-coded thresholds |
| 27 | `multi_channel_4` | 4-channel concurrent grid inference from four sources |
| 28 | `hand_landmark` | 21-point hand keypoint detection |

---

## Configuration

All demo settings live in `python_examples/config.yml`. Edit this file to change input source, model paths, or thresholds — no `.py` files need to be touched.

### Global Settings

```yaml
global:
  input_source: "video"       # webcam | video | image | rpicam
  video_path: "videos/sample.mp4"
  image_path: "videos/frame.jpg"
  webcam_index: 0             # USB camera index
  show_fps: true
  confidence_threshold: 0.5   # 0.0–1.0
  iou_threshold: 0.45
```

### Input Sources

#### USB webcam
```yaml
global:
  input_source: "webcam"
  webcam_index: 0             # find cameras with: ls /dev/video*
```

#### Video file
```yaml
global:
  input_source: "video"
  video_path: "videos/snowboard.mp4"   # .mp4 / .avi / .mov
```

#### Single image
```yaml
global:
  input_source: "image"
  image_path: "videos/frame.jpg"       # loops until 'q' is pressed
```

#### Raspberry Pi camera (rpicam)

Install the required library first:
```bash
sudo apt install -y python3-picamera2
```

Set `input_source` to `rpicam` in `config.yml`:
```yaml
global:
  input_source: "rpicam"
```

Verify camera is detected before use:
```bash
rpicam-hello --list-cameras
```

### Per-Demo Overrides

Each key under `demos:` overrides global values for that demo only. Unset fields fall back to the global default.

```yaml
demos:
  yolov8:
    model_path: "models/YoloV8N.dxnn"
    # all other settings inherited from global

  hand_landmark:
    model_path: "models/HandLandmark.dxnn"
    input_source: "webcam"    # always uses webcam regardless of global setting
```

### Advanced Demo Parameters

| Parameter | Demo | Type | Description |
|-----------|------|------|-------------|
| `polygon` | `trespassing` | `[[x,y], ...]` | Normalized coordinates of forbidden zone boundary |
| `line` | `smart_traffic` | `[x1,y1,x2,y2]` | Normalized counting line endpoints |
| `regions` | `store_queue_analysis` | list of polygons | Queue zone boundaries |
| `channels` | `multi_channel_4` | list of `{source, path}` | Per-channel video source |
| `wait_thresholds` | `store_queue_analysis` | `{green, yellow}` seconds | Color-code wait-time thresholds |
| `vehicle_classes` | `smart_traffic` | list of strings | Class names to count |
| `anchors` | non-Ultralytics models | list of `{stride, widths, heights}` | Anchor grid definitions |

All zone coordinates use normalized values in `[0.0, 1.0]` relative to frame dimensions. This makes configs resolution-independent:
- Top-left: `[0.0, 0.0]`
- Center: `[0.5, 0.5]`
- Bottom-right: `[1.0, 1.0]`

---

## Running Demos Directly

Every demo can be run without the launcher. CLI flags override the matching `config.yml` values.

```bash
# Use settings from config.yml
python python_examples/object_detection/yolov8_demo.py

# Override input source
python python_examples/object_detection/yolov8_demo.py --source video --path videos/sample.mp4
python python_examples/object_detection/yolov8_demo.py --source webcam
python python_examples/object_detection/yolov8_demo.py --source rpicam
python python_examples/object_detection/yolov8_demo.py --source image --path frame.jpg

# Override model and confidence
python python_examples/object_detection/yolov8_demo.py \
    --model models/YoloV8N.dxnn \
    --conf 0.3
```

Available CLI flags (same for all demos):

| Flag | Description |
|------|-------------|
| `--source` | Input source: `webcam`, `video`, `image`, `rpicam` |
| `--path` | Path to video or image file |
| `--model` | Path to `.dxnn` model file |
| `--conf` | Confidence threshold |
| `--iou` | NMS IoU threshold |
| `--labels` | Path to class label file (one label per line) |

---

## Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit demo |
| `↑` `↓` | Navigate launcher |
| `Enter` | Launch selected demo |

---

## Project Structure

```
python_examples/
├── start.sh              # Setup + launch (start here)
├── launcher.py           # Interactive TUI demo launcher
├── config.yml            # All demo settings
├── config_loader.py      # YAML config parser
├── requirements.txt      # Python dependencies
├── object_detection/     # YOLOv5–v12, v26, SCRFD, YOLOX demos
├── classification/       # YOLOv26cls demo
├── segmentation/         # DeepLabV3+, YOLOv8seg, YOLOv26seg demos
├── ppu/                  # Hardware-preprocessed variants
├── async_example/        # Async inference pattern
├── advanced/             # Zone, tracking, traffic, queue, multichannel demos
├── utils/                # Shared utilities (tracker, visualizer, fps counter, ...)
├── models/               # .dxnn compiled model files
└── videos/               # Input video files
    └── 360p/             # 360p videos for multi-channel demo
```

### Key Shared Utilities (`utils/`)

- **`runner.py`** — `run_demo()` manages the frame loop: open source → read frame → infer → draw → display → handle quit
- **`visualizer.py`** — drawing helpers for boxes, masks, keypoints, and labels
- **`tracker.py`** — simple ID-based object tracker used by `people_tracking` and other advanced demos
- **`fps.py`** — rolling FPS counter

---

## Troubleshooting

**No detections / zero bounding boxes**
- Lower `confidence_threshold` in `config.yml` (try `0.25`)
- Confirm the correct model is set for the demo under `demos:` in `config.yml`
- Check that the model file exists: `ls -lah resources/models/`

**Demo shows ○ (red dot) in launcher**
- The model file does not exist at the configured path
- Update `model_path` under the demo key in `config.yml`

**Camera not found**
- USB: run `ls /dev/video*` and update `webcam_index`
- libcamera: run `libcamera-hello --list-cameras`

**ImportError: No module named 'dxrt'**
- The venv is not active. Run: `source /opt/sixfab-dx/venv/bin/activate`
- Or use `./start.sh` which activates it automatically

**Low FPS**
- Use a PPU-enabled variant (e.g. `yolov5_ppu` instead of `yolov5`)
- Close other processes on the Raspberry Pi

---

*For issues or contributions, visit [sixfab/deepx-rpi5-examples](https://github.com/sixfab/deepx-rpi5-examples).*
