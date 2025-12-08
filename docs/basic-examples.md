# DEEPX Basic Examples Documentation

This document provides information about the basic AI examples available in the DEEPX RPi5 Examples repository.

## Table of Contents
- [Overview](#overview)
- [AI Examples](#ai-examples)

## Overview

The basic examples demonstrate core AI capabilities on Raspberry Pi 5. Each example supports:

- Real-time inference
- Multiple input sources
- Extensible architecture

### Supported Input Sources
- **Raspberry Pi Camera**: Built-in camera module
- **USB Cameras**: External webcams
- **Video Files**: MP4, AVI, MOV formats
- **Image Sequences**: Directory of images
- **RTSP Streams**: Network cameras (advanced)

### Output Options
- **Real-time Display**: OpenCV window
- **Video Recording**: Save processed video
- **Image Capture**: Save individual frames
- **Data Export**: JSON/CSV detection results

## AI Examples

### 1. ImageNet Classification (`classification/imagenet-classification.py`)

High-performance image classification using EfficientNetB0 model for 1000 ImageNet classes.

#### Features
- EfficientNetB0 model with DEEPX NPU acceleration
- 1000-class ImageNet dataset support
- Real-time camera input processing
- Batch processing capabilities
- Grid visualization for multiple images
- Accuracy tracking and statistics

#### Usage
```bash
# Basic camera usage
python basic_examples/classification/imagenet-classification.py --rpicam

# USB camera with specific index
python basic_examples/classification/imagenet-classification.py --usbcam --camera_index 0

# Single image classification
python basic_examples/classification/imagenet-classification.py --image path/to/image.jpg

# Configuration file usage
python basic_examples/classification/imagenet-classification.py --config basic_examples/src/configs/imagenet_classification.json

# Batch processing with custom model
python basic_examples/classification/imagenet-classification.py --model basic_examples/src/models/EfficientNetB0_4.dxnn
```

#### Model Specifications
- **Input Size**: 224×224 RGB
- **Classes**: 1000 ImageNet categories
- **Model File**: `EfficientNetB0_4.dxnn`
- **Performance**: 35+ FPS on RPi5

### 2. YOLO Object Detection (`object-detection/yolo-object-detection.py`)

Real-time object detection using YOLOv8/YOLOv5 models with 80 COCO classes.

#### Features
- YOLOv8N/YOLOv5s model support
- 80-class COCO dataset detection
- Real-time bounding box visualization
- Configurable confidence and IoU thresholds
- Multiple input sources support
- Non-Maximum Suppression (NMS)

#### Usage
```bash
# Raspberry Pi camera detection
python basic_examples/object-detection/yolo-object-detection.py --rpicam

# USB camera with custom settings
python basic_examples/object-detection/yolo-object-detection.py --usbcam --camera_index 0

# Video file processing
python basic_examples/object-detection/yolo-object-detection.py --video path/to/video.mp4

# Image detection
python basic_examples/object-detection/yolo-object-detection.py --image path/to/image.jpg

# Custom model and thresholds
python basic_examples/object-detection/yolo-object-detection.py --model basic_examples/src/models/YoloV8N.dxnn --confidence 0.4
```

#### Command Line Options
```bash
--camera SOURCE       Camera source (rpicam/usb)
--camera_index INT    USB camera device index (default: 0)
--video PATH          Video file path for processing
--image PATH          Single image file path
--model PATH          Custom DEEPX model file
--confidence FLOAT    Confidence threshold (default: 0.3)
--iou-threshold FLOAT IoU threshold for NMS (default: 0.4)
--show-fps            Display FPS counter
```

#### Model Specifications
- **Input Size**: 640×640 RGB
- **Classes**: 80 COCO categories
- **Model Files**: `YoloV8N.dxnn`, `YOLOV5S-1.dxnn`
- **Performance**: 30+ FPS on RPi5

### 3. DnCNN Image Denoiser (`denoiser/dncnn-denoiser.py`)

Deep learning-based image denoising using DnCNN architecture for real-time noise reduction.

#### Features
- DnCNN-2 model for Gaussian noise removal
- Real-time camera denoising
- Configurable noise simulation
- Before/after comparison display
- Multiple input format support

#### Usage
```bash
# Real-time camera denoising
python basic_examples/denoiser/dncnn-denoiser.py --rpicam --noise-std 15.0

# USB camera with custom noise level
python basic_examples/denoiser/dncnn-denoiser.py --usbcam --noise-std 20.0

# Image file denoising
python basic_examples/denoiser/dncnn-denoiser.py --image noisy_image.jpg

# Video denoising
python basic_examples/denoiser/dncnn-denoiser.py --video noisy_video.mp4
```

#### Model Specifications
- **Input Size**: 512×512 RGB
- **Model File**: `DnCNN-2.dxnn`
- **Noise Type**: Gaussian noise removal
- **Performance**: 25+ FPS on RPi5

### 4. YOLO Pose Estimation (`pose-estimation/yolo-pose-estimation.py`)

Human pose detection with 17 COCO keypoints using YOLOv5-pose architecture.

#### Features
- 17-keypoint human skeleton detection
- Multi-person pose tracking
- Real-time pose visualization
- Configurable confidence thresholds
- Skeleton connection drawing

#### Usage
```bash
# Real-time pose detection
python basic_examples/pose-estimation/yolo-pose-estimation.py --rpicam

# USB camera pose estimation
python basic_examples/pose-estimation/yolo-pose-estimation.py --usbcam

# Video file processing
python basic_examples/pose-estimation/yolo-pose-estimation.py --video dance_video.mp4

# Single image pose detection
python basic_examples/pose-estimation/yolo-pose-estimation.py --image person.jpg

# Custom confidence threshold
python basic_examples/pose-estimation/yolo-pose-estimation.py --rpicam --confidence 0.4
```

#### Keypoint Structure
- **Body Parts**: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles
- **Connections**: 16 skeleton connections for pose visualization
- **Model File**: `YOLOV5Pose640_1.dxnn`
- **Performance**: 25+ FPS on RPi5

### 5. Semantic Segmentation (`segmentation/segmentation.py`)

Pixel-level semantic segmentation using DeepLabV3+ MobileNetV2 for scene understanding.

#### Features
- DeepLabV3+ MobileNetV2 architecture
- 19-class Cityscapes or 3-class custom segmentation
- Real-time pixel-level classification
- Color-coded class visualization
- Multiple input source support

#### Usage
```bash
# Real-time segmentation (19 classes)
python basic_examples/segmentation/segmentation.py --rpicam --classes 19

# USB camera segmentation (3 classes)
python basic_examples/segmentation/segmentation.py --usbcam --classes 3

# Video file segmentation
python basic_examples/segmentation/segmentation.py --video street_video.mp4

# Image segmentation
python basic_examples/segmentation/segmentation.py --image street_scene.jpg
```

#### Segmentation Classes
**19-Class Mode (Cityscapes)**:
- road, sidewalk, building, wall, fence, pole, traffic light, traffic sign
- vegetation, terrain, sky, person, rider, car, truck, bus, train, motorcycle, bicycle

**3-Class Mode**:
- background, foot, body

#### Model Specifications
- **Input Size**: Variable (auto-resize)
- **Model File**: `DeepLabV3PlusMobileNetV2_2.dxnn`
- **Performance**: 20+ FPS on RPi5

### 6. Object Detection + Segmentation (`segmentation/object-det-and-seg.py`)

Combined object detection and semantic segmentation pipeline for comprehensive scene analysis.

#### Features
- Dual-pipeline architecture (detection + segmentation)
- Real-time bounding boxes and pixel masks
- Multi-threading for optimal performance
- Combined visualization overlay
- Configurable model selection

#### Usage
```bash
# Combined detection and segmentation
python basic_examples/segmentation/object-det-and-seg.py --rpicam

# USB camera with custom models
python basic_examples/segmentation/object-det-and-seg.py --usbcam --detection-model src/models/YoloV8N.dxnn --segmentation-model src/models/DeepLabV3PlusMobileNetV2_2.dxnn

# Video processing
python basic_examples/segmentation/object-det-and-seg.py --video input_video.mp4

# Performance optimized mode
python basic_examples/segmentation/object-det-and-seg.py --rpicam --frame-buffers 15
```

#### Pipeline Features
- **Parallel Processing**: Detection and segmentation run simultaneously
- **Frame Buffering**: Optimized memory management
- **Dual Visualization**: Bounding boxes + segmentation masks
- **Performance**: 15-20 FPS combined pipeline

## Pipeline Architecture

### Data Flow
1. **Input Capture**: Camera/file → raw frame
2. **Preprocessing**: Resize, normalize, format conversion
3. **Inference**: Model processing
4. **Postprocessing**: NMS, coordinate scaling, filtering
5. **Visualization**: Draw results
6. **Output**: Display/save processed frame

### Available Models

> **Note:** Models should be downloaded from the [releases page](https://github.com/sixfab/deepx-rpi5-examples/releases) and placed in the `basic_examples/src/models/` directory. Alternatively, you can download any compatible models from the [DEEPX Model Zoo](https://developer.deepx.ai/wp-content/modelzoo/model_zoo_fin.html) and use them in the models directory.

| Model File | Purpose | Input Size | Classes | Performance |
|------------|---------|------------|---------|-------------|
| `EfficientNetB0_4.dxnn` | Classification | 224×224 | 1000 | 35+ FPS |
| `YoloV8N.dxnn` | Object Detection | 640×640 | 80 | 30+ FPS |
| `YOLOV5S-1.dxnn` | Object Detection | 640×640 | 80 | 30+ FPS |
| `DnCNN-2.dxnn` | Denoising | 512×512 | N/A | 25+ FPS |
| `YOLOV5Pose640_1.dxnn` | Pose Estimation | 640×640 | 17 keypoints | 25+ FPS |
| `DeepLabV3PlusMobileNetV2_2.dxnn` | Segmentation | Variable | 19/3 classes | 20+ FPS |

## Performance Optimization

### Model Selection
- **YOLOv5s**: Fastest, good accuracy
- **YOLOv5m**: Balanced speed/accuracy
- **YOLOv5l**: Higher accuracy, slower

### Input Resolution
- **320x320**: Fastest inference
- **640x640**: Standard resolution
- **1280x1280**: High accuracy

### Expected Performance on RPi5
| Model | Resolution | FPS | Use Case |
|-------|------------|-----|---------|
| EfficientNetB0 | 224×224 | 35+ | Classification |
| YOLOv8N | 640×640 | 30+ | Object Detection |
| YOLOv5s | 640×640 | 30+ | Object Detection |
| DnCNN | 512×512 | 25+ | Denoising |
| YOLOv5-Pose | 640×640 | 25+ | Pose Estimation |
| DeepLabV3+ | 512×512 | 20+ | Segmentation |
| Combined Pipeline | 640×640 | 15-20 | Detection+Segmentation |

## Advanced Usage

### Interactive Controls

All camera-based examples support these keyboard controls during runtime:

- **'q'** or **ESC**: Quit application
- **'space'**: Pause/Resume processing
- **'s'**: Toggle statistics/FPS display
- **'r'**: Reset tracking (where applicable)
- **'c'**: Capture current frame (save as image)

### Common Command Line Arguments

All examples support these standard arguments:

```bash
# Camera Selection
--rpicam          # Use Raspberry Pi camera
--usbcam             # Use USB camera (auto-detect)
--camera_index 0         # Specific USB camera index

# File Processing
--image path/to/file.jpg    # Process single image
--video path/to/file.mp4    # Process video file

# Model Selection
--model path/to/model.dxnn  # Custom model path

# Performance Tuning
--confidence 0.3         # Confidence threshold (0.0-1.0)
--iou-threshold 0.4      # IoU threshold for NMS
--show-fps              # Display FPS counter
--resolution WxH        # Camera resolution (e.g., 640x480)

# Output Options
--save-output file.mp4   # Save processed video
--output-dir ./results   # Output directory for results
```

### Custom Models

#### 1. Using Your Own Models
```bash
# Use custom classification model
python basic_examples/classification/imagenet-classification.py --model /path/to/your_model.dxnn

# Use custom detection model
python basic_examples/object-detection/yolo-object-detection.py --model /path/to/your_yolo.dxnn
```

#### 2. Model Requirements
- **Format**: `.dxnn`
- **Input Format**: RGB for most models

### Integration Examples

#### 1. Flask Web Server
```python
from flask import Flask, Response
import cv2

app = Flask(__name__)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')
```

#### 2. MQTT Integration
```python
import paho.mqtt.client as mqtt

def publish_detections(detections):
    client = mqtt.Client()
    client.connect("localhost", 1883, 60)
    client.publish("deepx/detections", json.dumps(detections))
```

### Configuration Files

The examples use JSON configuration files located in `basic_examples/src/configs/`:

- **`imagenet_classification.json`**: Classification model config
- **`yolov8n_camera.json`**: YOLOv8 detection config  
- **`yolov8n_image.json`**: YOLOv8 image processing config
- **`yolov5s_image.json`**: YOLOv5 image processing config

## Quick Start Guide

### Prerequisites
```bash
# Install required dependencies
pip install opencv-python numpy torch torchvision ultralytics packaging

# For Raspberry Pi camera support
sudo apt install -y python3-picamera2
```

### Running Your First Example

1. **Object Detection (Recommended starting point)**:
```bash
cd /path/to/deepx-rpi5-examples
python basic_examples/object-detection/yolo-object-detection.py --rpicam
```

2. **Image Classification**:
```bash
python basic_examples/classification/imagenet-classification.py --rpicam
```

3. **Processing a Video File**:
```bash
python basic_examples/object-detection/yolo-object-detection.py --video /path/to/your/video.mp4
```

### Example Workflows

#### Batch Image Processing
```bash
# Process multiple images in a directory
for img in /path/to/images/*.jpg; do
    python basic_examples/classification/imagenet-classification.py --image "$img"
done
```

#### Performance Benchmarking
```bash
# Test different models and compare performance
python basic_examples/object-detection/yolo-object-detection.py --rpicam --show-fps
python basic_examples/classification/imagenet-classification.py --rpicam --show-fps
```

## Contributing

To add new examples:

1. Create new Python file in `basic_examples/` subdirectory
2. Follow existing code structure and naming conventions
3. Include proper error handling and camera setup
4. Add configuration support via JSON files (if needed)
5. Include comprehensive documentation and usage examples
6. Test on Raspberry Pi 5 with DEEPX NPU
7. Ensure compatibility with both RPi camera and USB cameras

---

*This documentation is maintained by the Sixfab community. For updates and contributions, please visit the GitHub repository.*