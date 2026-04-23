# C++ Examples

Standalone, idiomatic C++17 demos for the DEEPX NPU — mirroring the Python examples but built with only OpenCV and the DeepX SDK (`dxrt`). No Qt. No YAML. Each demo is a single readable file.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Building](#building)
- [The Launcher](#the-launcher)
- [Available Demos](#available-demos)
- [Running with a Custom Source](#running-with-a-custom-source)
- [Directory Layout](#directory-layout)
- [Adding a New Demo](#adding-a-new-demo)
- [Demo Anatomy](#demo-anatomy)

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Raspberry Pi 5 (or Linux + DeepX-compatible NPU) | — |
| DEEPX runtime (`sixfab-dx` APT package) | Provides `libdxrt.so` and headers at `/usr/local/include/dxrt/` |
| OpenCV 4.x dev headers | Installed by `setup.sh` |
| C++17 compiler, CMake ≥ 3.14, `pkg-config` | Installed by `setup.sh` |

---

## Building

Running `auto-install.sh` from the repo root builds the C++ examples automatically. To rebuild manually:

```bash
cd cpp_examples
./setup.sh
```

`setup.sh` installs system packages, runs CMake in Release mode, and builds all demos with `make -j$(nproc)`. Binaries land in `cpp_examples/build/`.

To rebuild after changing source files without reinstalling system packages:

```bash
cd cpp_examples/build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

---

## The Launcher

An interactive terminal UI lists all built demos, shows their status (built / not built), and launches the selected binary with the correct config and resource paths.

```bash
bash cpp_examples/start.sh
```

- Navigate with `↑` / `↓` (also `PageUp` / `PageDown`)
- Press `Enter` to launch
- Press `r` to rebuild
- Press `q` to quit
- Green `●` = binary found, Red `●` = binary not found (needs build)

---

## Available Demos

### Object Detection

| Binary | Description |
|--------|-------------|
| `object_detection/yolov8_demo` | YOLOv8 anchor-free detection (PPU) |
| `object_detection/yolov5_demo` | YOLOv5 anchor-based detection (PPU) |
| `object_detection/yolov11_demo` | YOLOv11 detection |

### Pose Estimation

| Binary | Description |
|--------|-------------|
| `pose_estimation/yolov5pose_demo` | 17-keypoint body pose (YOLOv5-Pose) |
| `pose_estimation/yolov26pose_demo` | Body pose with YOLOv26-Pose |
| `pose_estimation/hand_landmark_demo` | 21-point hand keypoint detection |

### Face

| Binary | Description |
|--------|-------------|
| `face/scrfd_demo` | Face detection with SCRFD |
| `face/emotion_demo` | Face detection + emotion classification |

### Segmentation

| Binary | Description |
|--------|-------------|
| `segmentation/yolov8seg_demo` | Instance segmentation (YOLOv8-seg) |
| `segmentation/yolov26seg_demo` | Instance segmentation (YOLOv26-seg) |
| `segmentation/deeplabv3_demo` | Semantic segmentation (DeepLabV3+) |

### Classification

| Binary | Description |
|--------|-------------|
| `classification/efficientnet_demo` | 1000-class ImageNet classification (EfficientNet) |
| `classification/mobilenet_demo` | MobileNetV2 classification |

### PPU — Hardware-Preprocessed Variants

| Binary | Description |
|--------|-------------|
| `ppu/scrfd_ppu_demo` | SCRFD face detection (PPU) |
| `ppu/yolov5_ppu_demo` | YOLOv5 detection (PPU) |
| `ppu/yolov5pose_ppu_demo` | YOLOv5 pose estimation (PPU) |
| `ppu/yolov7_ppu_demo` | YOLOv7 detection (PPU) |

### Async Inference

| Binary | Description |
|--------|-------------|
| `async_example/yolov8_async_demo` | Async pipeline using `RunAsync()` / `Wait()` |

### Advanced Analytics

| Binary | Description |
|--------|-------------|
| `advanced/people_tracking_demo` | Person tracking with persistent IDs |
| `advanced/trespassing_demo` | Forbidden-zone intrusion detection |
| `advanced/smart_traffic_demo` | Vehicle line-crossing counter |
| `advanced/store_queue_analysis_demo` | Queue wait-time analyzer |
| `advanced/multi_channel_4_demo` | 4-channel concurrent grid inference |
| `advanced/osnet_reid_demo` | Person re-identification with OSNet |

---

## Running with a Custom Source

Each demo reads its defaults from `cpp_examples/configs/<demo_name>.json` then applies CLI overrides:

```
--model <path>          Path to .dxnn model file
--source <type>         video | camera | libcamera | rtsp | image
--path <path>           Path to video / image file, or RTSP URL
--labels <path>         One label per line
--conf <float>          Score threshold (default 0.3)
--iou <float>           NMS IoU threshold (default 0.45)
--camera-index <n>      USB camera index (for --source camera)
--window <title>        OpenCV window title
--no-fps                Hide FPS overlay
```

Examples:

```bash
# USB webcam
./cpp_examples/build/object_detection/yolov8_demo --source camera --camera-index 0

# Raspberry Pi camera
./cpp_examples/build/object_detection/yolov8_demo --source libcamera --path 0:1536:864:30

# Video file with custom confidence threshold
./cpp_examples/build/object_detection/yolov8_demo --source video --path my_clip.mp4 --conf 0.4

# Single image
./cpp_examples/build/object_detection/yolov8_demo --source image --path frame.jpg
```

---

## Directory Layout

```
cpp_examples/
├── CMakeLists.txt          # Top-level build: common lib + demo subdirs
├── setup.sh                # One-shot system install + build
├── start.sh                # Interactive TUI launcher
├── configs/                # Per-demo JSON config files
├── common/                 # Shared library (sdk, ppu, input, vis, runner, ...)
├── object_detection/       # YOLO demos
├── pose_estimation/        # Pose demos
├── face/                   # Face detection and emotion
├── segmentation/           # Segmentation demos
├── classification/         # Classification demos
├── ppu/                    # Hardware-preprocessed variants
├── async_example/          # Async inference demo
├── advanced/               # Tracking, intrusion, traffic, queue, multi-channel
├── models/                 # .dxnn model files (downloaded by setup.sh)
└── videos/                 # Input video files (downloaded by setup.sh)
```

### Shared Library (`common/`)

| Module | Purpose |
|--------|---------|
| `sdk.*` | Thin wrappers around `dxrt` — load engine, run inference, get output tensors |
| `ppu.*` | Decode raw NPU output: `decodeYolov8Float()`, `decodeYolov5()`, etc. |
| `input.*` | `InputSource` abstraction — opens camera, video, image, or RTSP uniformly |
| `vis.*` | Draw detections, keypoints, masks on a cv::Mat |
| `runner.*` | `runDemo()` frame loop — ties InputSource, model, and draw callback together |

---

## Adding a New Demo

1. Place `<demo_name>.cpp` in the appropriate category directory (`object_detection/`, `pose_estimation/`, etc.)
2. Add it to that directory's `CMakeLists.txt`:
   ```cmake
   add_demo(<demo_name> <demo_name>.cpp)
   ```
3. Optionally add `configs/<demo_name>.json` with sensible defaults
4. Rebuild:
   ```bash
   cd cpp_examples/build
   make -j$(nproc)
   ```

---

## Demo Anatomy

Every demo follows the same three-part pattern:

```cpp
// 1. Model class — wraps engine load and inference
class YOLOv8Detector {
public:
    YOLOv8Detector(const Config& cfg);       // loads .dxnn, caches input dims
    std::vector<Detection> infer(cv::Mat bgr); // letterbox → RunAsync/Wait → decode → return
};

// 2. Draw callback — annotates a frame in place
auto draw = [&](cv::Mat& frame, const std::vector<Detection>& dets) {
    vis::drawDetections(frame, dets, labels);
};

// 3. main() — wires InputSource into runDemo()
int main(int argc, char** argv) {
    Config cfg = loadConfig("configs/yolov8_demo.json", argc, argv);
    auto labels = loadLabels(cfg.labels_path);
    YOLOv8Detector model(cfg);
    InputSource src(cfg);
    runDemo(src, model, draw);   // frame loop lives here
}
```

See any existing demo file for a complete working example.

---

*For issues or contributions, visit [sixfab/deepx-rpi5-examples](https://github.com/sixfab/deepx-rpi5-examples).*
