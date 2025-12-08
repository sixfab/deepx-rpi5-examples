# DEEPX Official Examples

This repository contains official demonstration examples for the DEEPX Runtime (DXRT) framework on Raspberry Pi 5. These examples showcase various artificial intelligence and machine learning capabilities including object detection, classification, pose estimation, and segmentation.

## Available Demonstrations

### 1. Object Detection
- **YOLO Single Channel** (`yolo_1channel.cpp`) - Basic YOLO object detection implementation
- **YOLO Multi-Channel** (`yolo_multi_channels.cpp`) - Multi-stream object detection
- **YOLO NXP** (`yolo_1channel_nxp.cpp`) - NXP platform-optimized implementation
- **Generic Object Detection** (`od.cpp`) - General-purpose object detection framework

### 2. Classification
- **Synchronous Classification** (`classification_sync.cpp`) - Blocking inference implementation
- **Asynchronous Classification** (`classification_async.cpp`) - Non-blocking inference implementation
- **ImageNet Classification** (`imagenet_classification.cpp`) - Standard ImageNet models
- **ImageNet Demo** (`imagenet_classification_demo.cpp`) - Interactive demonstration application

### 3. Pose Estimation
Human pose estimation for detecting body keypoints and skeleton structure.

### 4. Object Detection and Segmentation
Combined detection and instance segmentation in a unified pipeline.

### 5. Segmentation
Semantic segmentation for pixel-level scene understanding.

### 6. Denoiser
Image denoising utilizing deep learning models.

## Building the Project

### Prerequisites

- CMake version 3.14 or higher
- DEEPX Runtime (DXRT) installation

### Build Instructions

#### Native Build

```bash
# Create build directory
mkdir build && cd build

# Configure
cmake ..

# Build all demos
cmake --build .
```

### Build Targets

The project includes the following build targets:
- `object_detection` - Object detection demonstrations
- `classification` - Classification demonstrations
- `pose_estimation` - Pose estimation demonstrations
- `object_det_and_seg` - Combined detection and segmentation
- `segmentation` - Segmentation demonstrations

## Execution Instructions

After building, executables will be located in the build directory under their respective demonstration folders:

```bash
# Example: Run YOLO single channel demo
./build/demos/object_detection/yolo --help
```

Each demonstration may require:
- Pre-trained model files (`.dxnn` format)
- Input data (video files, images, or camera access)
- Configuration files

Refer to individual demonstration source files for specific usage instructions and command-line parameters.


---

**Note**: This repository contains demonstration applications for educational and development purposes. Production deployments require additional optimization, comprehensive error handling, and appropriate security considerations.
