# Sixfab | DeepX Python Examples

A collection of Python inference demos for the DeepX NPU running on the
DeepX NPU module with a Raspberry Pi host. Each demo loads a compiled
`.dxnn` model through the DeepX runtime (`dx_engine`) and streams results
from a webcam, video file, single image, or the Raspberry Pi camera.

The demos cover the most common computer-vision tasks: object detection,
image classification, semantic and instance segmentation, pose estimation,
object tracking, zone-based analytics (intrusion detection, queue timing,
line counting), multi-channel concurrent inference, and asynchronous
inference patterns.


## Quick Start

### Option A — One command (easiest)

Open a terminal on your Raspberry Pi and paste this single line:

```bash
curl -fsSL https://raw.githubusercontent.com/sixfab/rpi5-python-examples/main/install.sh | bash
```

That is it. The command clones the repository, installs the DeepX SDK,
sets up the Python environment, downloads the models and videos, and
opens the demo launcher — all in one go. No other steps needed.

### Option B — Clone and run

If you already have the repository or prefer to clone manually:

```bash
git clone https://github.com/sixfab/rpi5-python-examples.git
cd rpi5-python-examples
chmod +x start.sh
./start.sh
```

### What happens behind the scenes

Both options above run `start.sh`, which works through these steps in order:

| Step | What it does | Skipped if... |
|------|-------------|---------------|
| 1 | Installs `sixfab-dx` (DeepX SDK) via APT | already installed |
| 2 | Locates the Python venv at `/opt/sixfab-dx/venv` | always runs |
| 3 | Installs Python deps from `requirements.txt` | always runs (fast) |
| 4 | Downloads `models.tar.gz` and `videos.tar.gz` | directory not empty |
| 5 | Launches the demo menu | — |

On subsequent runs just call `./start.sh` from the project folder — it
goes straight to the launcher since everything is already set up.

> If you prefer to do things manually, see the [Manual Setup](#manual-setup) section below.


## Hardware Requirements

| Component | Details |
|---|---|
| Raspberry Pi | Tested on Raspberry Pi 5 |
| DeepX NPU | Required for all inference |
| RPi Camera Module | Optional — for libcamera input source |
| USB Webcam | Optional — for webcam input source |


## Installing the DeepX Runtime

The DeepX runtime is distributed through the Sixfab APT repository. The
steps below add the repository, install the `sixfab-dx` package (which
ships the runtime, Python wheels, and a pre-built virtual environment),
and verify that the NPU is detected.

**Step 1 — Add the GPG key**

```bash
wget -qO - https://sixfab.github.io/sixfab_dx/public.gpg | sudo gpg --dearmor -o /usr/share/keyrings/sixfab-dx.gpg
```

**Step 2 — Add the APT repository**

```bash
echo "deb [signed-by=/usr/share/keyrings/sixfab-dx.gpg] https://sixfab.github.io/sixfab_dx trixie main" | sudo tee /etc/apt/sources.list.d/sixfab-dx.list
```

**Step 3 — Install the package**

```bash
sudo apt update && sudo apt install sixfab-dx
```

**Step 4 — Verify installation**

```bash
dxrt-cli -s
```

The command prints the NPU status. If the device is recognized the SDK
is ready to use.

**Step 5 — Python environment**

> **Option A — Use the pre-built venv (recommended)**
>
> ```bash
> source /opt/sixfab-dx/venv/bin/activate
> ```

> **Option B — Install wheels into your own venv**
>
> ```bash
> # Install numpy wheel first (required dependency)
> # Then install the DX engine
> # For python 3.13
> pip install /opt/sixfab-dx/wheels/numpy-2.4.4-cp313-cp313-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl
> pip install /opt/sixfab-dx/wheels/dx_engine-3.3.0-cp313-cp313-linux_aarch64.whl
> # For python 3.11
> pip install /opt/sixfab-dx/wheels/numpy-2.4.4-cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl
> pip install /opt/sixfab-dx/wheels/dx_engine-3.3.0-cp311-cp311-linux_aarch64.whl
> ```

> All commands in this README assume the venv is activated.

<details>
<summary>Updating sixfab-dx</summary>

```bash
sudo apt update && sudo apt upgrade sixfab-dx
```
</details>

<details>
<summary>Uninstalling sixfab-dx</summary>

```bash
sudo apt remove sixfab-dx
sudo rm /etc/apt/sources.list.d/sixfab-dx.list
sudo rm /usr/share/keyrings/sixfab-dx.gpg
sudo apt update
```
</details>


## Manual Setup

If you prefer not to use `start.sh`, follow these steps:

```bash
# 1. Clone
git clone https://github.com/sixfab/rpi5-python-examples.git
cd rpi5-python-examples

# 2. Activate the SDK venv
source /opt/sixfab-dx/venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Download models and videos
chmod +x download.sh
./download.sh

# 5. Launch
python launcher.py
```


## The Launcher

`launcher.py` is a keyboard-driven TUI built with the Rich
library. It lists every demo, shows model status at a glance, and launches
the selected demo using the settings from `config.yml`.

- Navigate with the `↑` and `↓` arrow keys.
- Press `Enter` to launch the highlighted demo.
- Press `q` to quit.
- A green `●` next to a demo name means the model file was found on disk.
- A red `○` means the model file is missing. Update `model_path` for
  that demo in `config.yml`.
- Demos launch instantly with the settings from `config.yml`. There are
  no interactive prompts.

### Available demos

| # | Demo | Category | Description |
|---|------|----------|-------------|
| 1  | scrfd                | Object Detection | Face detection with SCRFD |
| 2  | yolov10              | Object Detection | General object detection (YOLOv10) |
| 3  | yolov11              | Object Detection | General object detection (YOLOv11) |
| 4  | yolov12              | Object Detection | General object detection (YOLOv12) |
| 5  | yolov26              | Object Detection | General object detection (YOLOv26) |
| 6  | yolov26pose          | Object Detection | Human pose estimation (skeleton keypoints) |
| 7  | yolov5               | Object Detection | General object detection (YOLOv5) |
| 8  | yolov5face           | Object Detection | Face detection with YOLOv5-face |
| 9  | yolov5pose           | Object Detection | Human pose estimation (skeleton keypoints) |
| 10 | yolov7               | Object Detection | General object detection (YOLOv7) |
| 11 | yolov8               | Object Detection | General object detection (YOLOv8) |
| 12 | yolov9               | Object Detection | General object detection (YOLOv9) |
| 13 | yolox                | Object Detection | General object detection (YOLOX) |
| 14 | yolov26cls           | Classification   | Image classification with YOLOv26-cls |
| 15 | deeplabv3            | Segmentation     | Semantic segmentation with DeepLabV3+ |
| 16 | yolov26seg           | Segmentation     | Instance segmentation with YOLOv26-seg |
| 17 | yolov8seg            | Segmentation     | Instance segmentation with YOLOv8-seg |
| 18 | scrfd_ppu            | PPU              | SCRFD face detection (hardware-preprocessed) |
| 19 | yolov5_ppu           | PPU              | YOLOv5 detection (hardware-preprocessed) |
| 20 | yolov5pose_ppu       | PPU              | YOLOv5 pose estimation (hardware-preprocessed) |
| 21 | yolov7_ppu           | PPU              | YOLOv7 detection (hardware-preprocessed) |
| 22 | yolov8_async         | Async            | Async inference pipeline example |
| 23 | trespassing          | Advanced         | Forbidden-zone intrusion alert |
| 24 | people_tracking      | Advanced         | Person tracking with persistent IDs |
| 25 | smart_traffic        | Advanced         | Vehicle line-crossing counter |
| 26 | store_queue_analysis | Advanced         | Queue wait-time analyzer |
| 27 | multi_channel_4      | Advanced         | 4-channel concurrent grid inference |
| 28 | hand_landmark        | Advanced         | 21-point hand keypoint detection |


## Configuration

All demo settings live in `config.yml`. You never need to edit any `.py`
file.

### Global settings

```yaml
global:
  input_source: "video"       # webcam | video | image | rpicam
  video_path: "..."           # path to video file
  image_path: "..."           # path to image file
  webcam_index: 0             # USB camera index (0, 1, 2...)
  show_fps: true              # show FPS counter on screen
  confidence_threshold: 0.5   # detection confidence (0.0–1.0)
  iou_threshold: 0.45         # NMS overlap threshold
```

### Input sources

#### USB webcam

```yaml
global:
  input_source: "webcam"
  webcam_index: 0
```

> Find available cameras with `ls /dev/video*`. If you have more than one
> camera, try index `1`, `2`, and so on.

#### Video file

```yaml
global:
  input_source: "video"
  video_path: "videos/snowboard.mp4"
```

> Supported formats: `.mp4`, `.avi`, `.mov`.

#### Single image

```yaml
global:
  input_source: "image"
  image_path: "videos/frame.jpg"
```

> The image will loop — press `q` to quit.

#### Raspberry Pi camera (rpicam)

Install the required library first:

```bash
sudo apt install -y python3-picamera2

# or 

pip install picamera2```

Then set `input_source` to `rpicam` in `config.yml`:

```yaml
global:
  input_source: "rpicam"
```

> Verify the camera is detected before use:
>
> ```bash
> rpicam-hello --list-cameras
> ```
>
> Enable the camera interface if needed:
>
> ```bash
> sudo raspi-config  # → Interface Options → Camera → Enable
> ```

### Per-demo model paths

Every demo key under `demos:` overrides global values for that demo. Any
field not set under the demo key falls back to the global default.

```yaml
demos:
  yolov8:
    model_path: "models/YoloV8N.dxnn"
    # All other settings (confidence, input_source) inherited from global
```

A demo can also lock its own input source regardless of the global
setting:

```yaml
demos:
  hand_landmark:
    model_path: "..."
    input_source: "webcam"   # this demo always uses webcam regardless of global
```

### Advanced demo parameters

| Parameter | Demo | Type | Description |
|---|---|---|---|
| `polygon` | trespassing | list of `[x,y]` | Normalized points of forbidden zone |
| `line` | smart_traffic | `[x1,y1,x2,y2]` | Normalized counting line |
| `regions` | store_queue_analysis | list of polygons | Queue zone boundaries |
| `channels` | multi_channel_4 | list of `{source,path}` | Per-channel video source |
| `wait_thresholds` | store_queue_analysis | `{green,yellow}` seconds | Color-code wait times |
| `vehicle_classes` | smart_traffic | list of strings | Class names to count |
| `anchors` | non-ultralytics models | list of `{stride,widths,heights}` | Anchor definitions |

All zone coordinates (`polygon`, `line`, `regions`) use normalized values
in the range `0.0`–`1.0`, relative to the frame dimensions. This makes
configurations resolution-independent. For example, on a 1920×1080 frame:

- Top-left corner = `[0.0, 0.0]`
- Center = `[0.5, 0.5]`
- Bottom-right = `[1.0, 1.0]`


## Running Demos Directly

Every demo can also be run without the launcher. The CLI flags override
the matching values from `config.yml`. Using `yolov8` as the example:

```bash
# Use settings from config.yml
python object_detection/yolov8_demo.py

# Override input source via CLI
python object_detection/yolov8_demo.py --source video --path videos/snowboard.mp4
python object_detection/yolov8_demo.py --source webcam
python object_detection/yolov8_demo.py --source image --path frame.jpg
python object_detection/yolov8_demo.py --source rpicam

# Override model and confidence
python object_detection/yolov8_demo.py --model models/YoloV8N.dxnn --conf 0.3
```

| Flag | Default | Description |
|---|---|---|
| `--source` | from config.yml | Input source: `webcam`, `video`, `image`, `rpicam` |
| `--path` | from config.yml | Path to video or image file |
| `--model` | from config.yml | Path to `.dxnn` model file |
| `--conf` | from config.yml | Confidence threshold |
| `--iou` | from config.yml | NMS IoU threshold |
| `--labels` | from config.yml | Path to class label file |


## Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit the demo |
| `↑` `↓` | Navigate launcher menu |
| `Enter` | Launch selected demo |


## Troubleshooting

**No bounding boxes / zero detections**
- Lower `confidence_threshold` in `config.yml` (try `0.25`).
- Verify the correct model is set for the demo in `config.yml`.
- Check that the model file exists: `ls -lah models/`.

**Demo shows ○ (red dot) in launcher**
- The model file does not exist at the configured path.
- Update `model_path` under the demo key in `config.yml`.

**Camera not found**
- Webcam: run `ls /dev/video*` and update `webcam_index` in `config.yml`.
- rpicam: run `rpicam-hello --list-cameras` to verify detection.
- Try a different `webcam_index` (`0`, `1`, `2`...).

**ImportError: No module named 'dxrt'**
- The SDK venv is not activated.
- Run: `source /opt/sixfab-dx/venv/bin/activate`
- Or just use `./start.sh` which handles this automatically.

**download.sh fails**
- Verify internet connection.
- Check that the GitHub release exists and has the assets attached.
- Try running `./start.sh` again — it will retry the download.

**Low FPS / slow inference**
- Switch to a PPU-enabled model variant (e.g. `yolov8l-ppu` instead of `yolov8l`).
- Close other running processes on the Raspberry Pi.


## Project Structure

```
deepx-demos/
├── start.sh                 # One-shot setup and launch (start here)
├── download.sh              # Download models and videos only
├── launcher.py              # Interactive TUI demo launcher
├── config.yml               # All demo settings (edit this)
├── config_loader.py         # YAML config parser
├── requirements.txt
├── object_detection/        # YOLOv5–v12, v26, SCRFD, YOLOX demos
├── classification/          # YOLOv26cls demo
├── segmentation/            # DeepLabV3, YOLOv8seg, YOLOv26seg demos
├── ppu/                     # Hardware-preprocessed PPU variants
├── async_example/           # Async inference pattern
├── advanced/                # Zone, tracking, traffic, queue, multichannel demos
├── utils/                   # Shared utilities (tracker, visualizer, fps, etc.)
├── models/                  # .dxnn compiled model files
└── videos/                  # Input video files
    └── 360p/                # 360p video files for multi-channel demo
```
