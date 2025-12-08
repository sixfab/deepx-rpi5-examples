import cv2
import numpy as np
import argparse
import threading
import time
import sys
from queue import Queue
from typing import List, Tuple, Dict
from dx_engine import InferenceEngine, Configuration
from packaging import version
import torch
import torchvision
from ultralytics.utils import ops

# Try to import picamera2 for Raspberry Pi camera support
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    print("Warning: picamera2 not available. Raspberry Pi camera support disabled.")

# Color table for bounding boxes
COLOR_TABLE = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128),
    (128, 0, 255), (0, 128, 255), (192, 192, 192), (128, 128, 128)
]

# Segmentation configuration for 19 classes (Cityscapes)
SEGMENTATION_CONFIG = [
    {"id": 0, "name": "road", "color": (128, 64, 128)},
    {"id": 1, "name": "sidewalk", "color": (244, 35, 232)},
    {"id": 2, "name": "building", "color": (70, 70, 70)},
    {"id": 3, "name": "wall", "color": (102, 102, 156)},
    {"id": 4, "name": "fence", "color": (190, 153, 153)},
    {"id": 5, "name": "pole", "color": (153, 153, 153)},
    {"id": 6, "name": "traffic light", "color": (51, 255, 255)},
    {"id": 7, "name": "traffic sign", "color": (220, 220, 0)},
    {"id": 8, "name": "vegetation", "color": (107, 142, 35)},
    {"id": 9, "name": "terrain", "color": (152, 251, 152)},
    {"id": 10, "name": "sky", "color": (255, 0, 0)},
    {"id": 11, "name": "person", "color": (0, 51, 255)},
    {"id": 12, "name": "rider", "color": (255, 0, 0)},
    {"id": 13, "name": "car", "color": (255, 51, 0)},
    {"id": 14, "name": "truck", "color": (255, 51, 0)},
    {"id": 15, "name": "bus", "color": (255, 51, 0)},
    {"id": 16, "name": "train", "color": (0, 80, 100)},
    {"id": 17, "name": "motorcycle", "color": (0, 0, 230)},
    {"id": 18, "name": "bicycle", "color": (119, 11, 32)}
]

FRAME_BUFFERS = 15

def setup_camera(use_rpicam=False, camera_index=0, width=640, height=480, fps=30):
    """Setup camera based on the type requested."""
    if use_rpicam:
        if not PICAMERA2_AVAILABLE:
            print("Error: picamera2 is not installed. Please install it with:")
            print("  sudo apt install -y python3-picamera2")
            sys.exit(1)
        
        print(f"Initializing Raspberry Pi Camera...")
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        picam2.configure(config)
        picam2.start()
        
        def read_frame():
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return True, frame
        
        return picam2, read_frame
    else:
        print(f"Initializing USB Camera (index: {camera_index})...")
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        
        if not cap.isOpened():
            print(f"Error: Cannot open USB camera with index {camera_index}")
            sys.exit(1)
        
        def read_frame():
            return cap.read()
        
        return cap, read_frame

def letter_box(image_src, new_shape=(512, 512), fill_color=(114, 114, 114), format=None):
    """Apply letterbox resizing to image."""
    src_shape = image_src.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / src_shape[0], new_shape[1] / src_shape[1])
    ratio = r, r
    new_unpad = int(round(src_shape[1] * r)), int(round(src_shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    dw /= 2
    dh /= 2

    if src_shape[::-1] != new_unpad:
        image_src = cv2.resize(image_src, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image_new = cv2.copyMakeBorder(image_src, top, bottom, left, right, 
                                   cv2.BORDER_CONSTANT, value=fill_color)
    
    if format is not None:
        image_new = cv2.cvtColor(image_new, format)

    return image_new, ratio, (dw, dh)

class YoloParams:
    """YOLO model parameters"""
    def __init__(self, width=640, height=640, conf_thresh=0.25, iou_thresh=0.45, classes=None):
        self.width = width
        self.height = height
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.classes = classes or []
        self.colors = np.random.randint(0, 256, [len(self.classes), 3], np.uint8).tolist()

class OdSegmentationProcessor:
    def __init__(self, od_model_path, seg_model_path, od_params, num_classes=19):
        # Check DX-RT version
        if version.parse(Configuration().get_version()) < version.parse("3.0.0"):
            print("DX-RT version 3.0.0 or higher is required.")
            sys.exit(1)
        
        self.od_model_path = od_model_path
        self.seg_model_path = seg_model_path
        self.od_params = od_params
        self.num_classes = num_classes
        
        # Initialize engines
        print("Loading Object Detection model...")
        self.od_engine = InferenceEngine(od_model_path)
        
        print("Loading Segmentation model...")
        self.seg_engine = InferenceEngine(seg_model_path)
        
        # Check model version
        if version.parse(self.od_engine.get_model_version()) < version.parse('7'):
            print("Model version 7 or higher is required.")
            sys.exit(1)
        
        # Get input shapes
        od_input_size = int(np.sqrt(self.od_engine.get_input_size() / 3))
        self.od_input_shape = (od_input_size, od_input_size)
        
        seg_input_size = int(np.sqrt(self.seg_engine.get_input_size() / 3))
        self.seg_input_shape = (seg_input_size, seg_input_size)
        
        print(f"OD Input Shape: {self.od_input_shape}")
        print(f"Seg Input Shape: {self.seg_input_shape}")
        
        # Results storage
        self.frames = [None] * FRAME_BUFFERS
        self.seg_results = [None] * FRAME_BUFFERS
        self.od_results = [None] * FRAME_BUFFERS
        self.ratios = [None] * FRAME_BUFFERS
        self.offsets = [None] * FRAME_BUFFERS
        
        self.od_process_count = 0
        self.seg_process_count = 0
        self.lock = threading.Lock()
        
        self.display_start = False
        self.display_exit = False
        self.app_quit = False

    def postprocess_segmentation(self, output):
        """Postprocess segmentation output"""
        if isinstance(output, list):
            output = output[0]
        
        output = np.squeeze(output)
        
        # Handle different output formats
        if output.dtype == np.float32 or output.dtype == np.float16:
            # Check if output has class dimension
            if len(output.shape) == 3:
                # Format: [classes, height, width]
                seg_map = np.argmax(output, axis=0)
            elif len(output.shape) == 2:
                # Format: [height, width] - already segmentation map
                seg_map = output
            else:
                # Flatten and reshape if needed
                h, w = self.seg_input_shape
                if output.shape[0] == h * w:
                    seg_map = output.reshape(h, w)
                else:
                    # Try to infer dimensions
                    total_pixels = output.size
                    if total_pixels == h * w * self.num_classes:
                        output = output.reshape(self.num_classes, h, w)
                        seg_map = np.argmax(output, axis=0)
                    elif total_pixels == h * w:
                        seg_map = output.reshape(h, w)
                    else:
                        print(f"Warning: Unexpected segmentation output shape: {output.shape}")
                        seg_map = np.zeros((h, w), dtype=np.uint8)
        else:
            seg_map = output
        
        # Ensure seg_map is 2D
        if len(seg_map.shape) > 2:
            seg_map = seg_map.squeeze()
        
        # Get final dimensions
        if len(seg_map.shape) == 2:
            h, w = seg_map.shape
        else:
            h, w = self.seg_input_shape
            seg_map = np.zeros((h, w), dtype=np.uint8)
        
        # Create colored segmentation mask
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_info in SEGMENTATION_CONFIG:
            class_id = class_info["id"]
            color = class_info["color"]
            mask = seg_map == class_id
            if mask.shape == (h, w):
                colored_mask[mask] = color
        
        return colored_mask

    def postprocess_yolo(self, outputs, ratio, offset, original_shape):
        """Postprocess YOLO detection output"""
        if isinstance(outputs, list):
            output = outputs[0]
        else:
            output = outputs
        
        output = np.squeeze(output)
        
        # Handle different output formats
        if len(output.shape) == 2:
            x = output.T if output.shape[0] < output.shape[1] else output
        else:
            x = output
        
        # Apply confidence threshold
        if x.shape[1] > 5:  # Has class predictions
            box = ops.xywh2xyxy(x[..., :4])
            conf = np.max(x[..., 4:], axis=-1, keepdims=True)
            j = np.argmax(x[..., 4:], axis=-1, keepdims=True)
        else:
            box = ops.xywh2xyxy(x[..., :4])
            conf = x[..., 4:5]
            j = np.zeros_like(conf)
        
        mask = conf.flatten() > self.od_params.conf_thresh
        filtered = np.concatenate((box, conf, j.astype(np.float32)), axis=1)[mask]
        
        if len(filtered) == 0:
            return []
        
        # Sort by confidence
        sorted_indices = np.argsort(-filtered[:, 4])
        x = filtered[sorted_indices]
        x = torch.Tensor(x)
        
        # Apply NMS
        x = x[torchvision.ops.nms(x[:,:4], x[:, 4], self.od_params.iou_thresh)]
        
        # Transform boxes back to original image coordinates
        boxes = []
        for detection in x.numpy():
            pt1 = detection[0:2].astype(int)
            pt2 = detection[2:4].astype(int)
            conf = detection[4]
            label = int(detection[5])
            
            # Transform coordinates
            pt1, pt2 = self.transform_box(pt1, pt2, ratio, offset, original_shape)
            
            boxes.append({
                'bbox': [pt1[0], pt1[1], pt2[0], pt2[1]],
                'class_id': label,
                'confidence': float(conf)
            })
        
        print(f"[Result] Detected {len(boxes)} Boxes.")
        return boxes

    def transform_box(self, pt1, pt2, ratio, offset, original_shape):
        """Transform bounding box coordinates back to original image space"""
        dw, dh = offset
        pt1[0] = int((pt1[0] - dw) / ratio[0])
        pt1[1] = int((pt1[1] - dh) / ratio[1])
        pt2[0] = int((pt2[0] - dw) / ratio[0])
        pt2[1] = int((pt2[1] - dh) / ratio[1])

        pt1[0] = max(0, min(pt1[0], original_shape[1]))
        pt1[1] = max(0, min(pt1[1], original_shape[0]))
        pt2[0] = max(0, min(pt2[0], original_shape[1]))
        pt2[1] = max(0, min(pt2[1], original_shape[0]))

        return pt1, pt2

    def draw_results(self, img, boxes):
        """Draw bounding boxes on image"""
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box['bbox']
            class_id = box['class_id']
            confidence = box['confidence']
            
            color = tuple(map(int, self.od_params.colors[class_id % len(self.od_params.colors)]))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Get class name
            class_name = self.od_params.classes[class_id] if class_id < len(self.od_params.classes) else f"Class {class_id}"
            
            # Terminal output
            print(f"[{idx}] {class_name}: {confidence:.2f}, BBox: ({x1}, {y1}) -> ({x2}, {y2})")
            
            # Add label text with background
            label_text = f"{class_name} {confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
            
            text_offset_y = y1 - 10
            if text_offset_y < text_height + 10:
                text_offset_y = y1 + text_height + 10
            
            cv2.rectangle(img, 
                         (x1, text_offset_y - text_height - 5),
                         (x1 + text_width + 5, text_offset_y + baseline),
                         color, -1)
            
            cv2.putText(img, label_text, 
                       (x1 + 2, text_offset_y - 5),
                       font, font_scale, (255, 255, 255), thickness)
        
        return img

    def display_thread(self, fps_only=False):
        """Display results thread"""
        while not self.display_start:
            time.sleep(0.01)
        
        if not fps_only:
            cv2.namedWindow("OD + Segmentation", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("OD + Segmentation", 960, 540)
        
        print("Press 'q' to quit...")
        
        while not self.app_quit:
            if self.display_exit:
                break
            
            with self.lock:
                od_idx = max(0, (self.od_process_count - 1) % FRAME_BUFFERS)
                seg_idx = max(0, (self.seg_process_count - 1) % FRAME_BUFFERS)
            
            if self.seg_process_count > 0 and self.od_process_count > 0:
                if self.frames[od_idx] is None:
                    continue
                
                display = self.frames[od_idx].copy()
                
                # Overlay segmentation
                if self.seg_results[seg_idx] is not None:
                    seg_resized = cv2.resize(self.seg_results[seg_idx], 
                                            (display.shape[1], display.shape[0]))
                    display = cv2.addWeighted(display, 0.6, seg_resized, 0.4, 0)
                
                # Draw detections
                if self.od_results[od_idx] is not None:
                    display = self.draw_results(display, self.od_results[od_idx])
                
                if not fps_only:
                    cv2.imshow("OD + Segmentation", display)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.display_exit = True
                        self.app_quit = True
        
        cv2.destroyAllWindows()

    def process_frame(self, frame, index):
        """Process a single frame"""
        self.frames[index] = frame
        
        # Preprocess for object detection
        od_input, od_ratio, od_offset = letter_box(frame, self.od_input_shape, 
                                                    fill_color=(114, 114, 114), 
                                                    format=cv2.COLOR_BGR2RGB)
        
        # Preprocess for segmentation
        seg_input = cv2.resize(frame, (self.seg_input_shape[1], self.seg_input_shape[0]))
        
        # Store transformation parameters
        self.ratios[index] = od_ratio
        self.offsets[index] = od_offset
        
        try:
            # Run segmentation inference
            seg_output = self.seg_engine.run([seg_input])
            
            # Debug: Print output shape on first frame
            if index == 0:
                print(f"[DEBUG] Segmentation output shape: {[o.shape if hasattr(o, 'shape') else len(o) for o in seg_output]}")
            
            seg_result = self.postprocess_segmentation(seg_output)
            
            with self.lock:
                self.seg_results[index] = seg_result
                self.seg_process_count += 1
        except Exception as e:
            print(f"[WARNING] Segmentation failed: {e}")
            # Create empty segmentation result
            with self.lock:
                self.seg_results[index] = np.zeros((self.seg_input_shape[0], self.seg_input_shape[1], 3), dtype=np.uint8)
                self.seg_process_count += 1
        
        try:
            # Run object detection inference
            od_output = self.od_engine.run([od_input])
            
            # Debug: Print output shape on first frame
            if index == 0:
                print(f"[DEBUG] OD output shape: {[o.shape if hasattr(o, 'shape') else len(o) for o in od_output]}")
            
            od_result = self.postprocess_yolo(od_output, od_ratio, od_offset, frame.shape)
            
            with self.lock:
                self.od_results[index] = od_result
                self.od_process_count += 1
        except Exception as e:
            print(f"[WARNING] Object detection failed: {e}")
            with self.lock:
                self.od_results[index] = []
                self.od_process_count += 1

    def run_image(self, image_path, loop=False):
        """Process image file"""
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Cannot read image {image_path}")
            return
        
        loop_count = 100 if loop else 1
        print(f"Processing image {loop_count} times...")
        
        display_thread = threading.Thread(target=self.display_thread, args=(False,))
        display_thread.start()
        
        start_time = time.time()
        
        for i in range(loop_count):
            index = i % FRAME_BUFFERS
            self.process_frame(frame, index)
            
            if i == 0:
                self.display_start = True
        
        # Wait for processing to complete
        while self.seg_process_count < loop_count or self.od_process_count < loop_count:
            time.sleep(0.01)
        
        self.display_exit = True
        self.app_quit = True
        display_thread.join()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n[INFO] Total time: {total_time*1000000:.0f} us")
        print(f"[INFO] Per frame time: {total_time*1000000/loop_count:.0f} us")
        print(f"[INFO] FPS: {loop_count/total_time:.2f}")
        
        # Save result
        if self.od_results[0] is not None:
            result_img = self.frames[0].copy()
            if self.seg_results[0] is not None:
                seg_resized = cv2.resize(self.seg_results[0], 
                                        (result_img.shape[1], result_img.shape[0]))
                result_img = cv2.addWeighted(result_img, 0.6, seg_resized, 0.4, 0)
            result_img = self.draw_results(result_img, self.od_results[0])
            cv2.imwrite("result.jpg", result_img)
            print("Result saved to result.jpg")

    def run_video(self, video_path=None, use_rpicam=False, use_usbcam=False, 
                  camera_index=0, loop=False, fps_only=False, target_fps=0):
        """Process video file or camera"""
        camera_mode = use_rpicam or use_usbcam
        
        if camera_mode:
            camera, read_frame = setup_camera(use_rpicam=use_rpicam, camera_index=camera_index)
        else:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Cannot open video {video_path}")
                return
            
            print(f"VideoCapture FPS: {int(cap.get(cv2.CAP_PROP_FPS))}")
            print(f"VideoCapture Frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
            print(f"Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))} x {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
            read_frame = cap.read
        
        display_thread = threading.Thread(target=self.display_thread, args=(fps_only,))
        display_thread.start()
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = read_frame()
            
            if not ret:
                if loop and not camera_mode:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    if not camera_mode:
                        print("\nVideo ended.")
                    self.display_exit = True
                    self.app_quit = True
                    break
            
            index = frame_count % FRAME_BUFFERS
            self.process_frame(frame, index)
            
            frame_count += 1
            
            if frame_count == 1:
                self.display_start = True
            
            if self.app_quit:
                break
            
            # FPS control
            if target_fps > 0 and not camera_mode:
                elapsed = time.time() - start_time
                expected_time = frame_count / target_fps
                if elapsed < expected_time:
                    time.sleep(expected_time - elapsed)
        
        # Cleanup
        if camera_mode:
            if use_rpicam and PICAMERA2_AVAILABLE:
                camera.stop()
            else:
                camera.release()
        else:
            cap.release()
        
        # Wait for processing to complete
        while self.seg_process_count < frame_count or self.od_process_count < frame_count:
            time.sleep(0.01)
        
        self.display_exit = True
        self.app_quit = True
        display_thread.join()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n[INFO] Total time: {total_time*1000000:.0f} us")
        print(f"[INFO] Per frame time: {total_time*1000000/frame_count:.0f} us")
        print(f"[INFO] FPS: {frame_count/total_time:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Object Detection + Segmentation Demo")
    parser.add_argument("-m0", "--od_modelpath", required=True, help="Object detection model path")
    parser.add_argument("-m1", "--seg_modelpath", required=True, help="Segmentation model path")
    parser.add_argument("--classes", nargs='+', default=[], help="Class names for detection")
    parser.add_argument("--conf_thresh", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou_thresh", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("-i", "--image", help="Image file input")
    parser.add_argument("-v", "--video", help="Video file input")
    parser.add_argument("--rpicam", action="store_true", help="Use Raspberry Pi camera")
    parser.add_argument("--usbcam", action="store_true", help="Use USB camera")
    parser.add_argument("--camera_index", type=int, default=0, help="Camera device index")
    parser.add_argument("-l", "--loop", action="store_true", help="Loop video/image")
    parser.add_argument("--fps_only", action="store_true", help="FPS only, no visualization")
    parser.add_argument("--target_fps", type=int, default=0, help="Target FPS")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.rpicam and args.usbcam:
        print("Error: Cannot use both --rpicam and --usbcam at the same time")
        sys.exit(1)
    
    if not args.image and not args.video and not args.rpicam and not args.usbcam:
        print("Error: Please specify input source (--image, --video, --rpicam, or --usbcam)")
        sys.exit(1)
    
    # Load default classes if not provided
    if not args.classes:
        args.classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    
    # Create YOLO parameters
    od_params = YoloParams(
        width=640,
        height=640,
        conf_thresh=args.conf_thresh,
        iou_thresh=args.iou_thresh,
        classes=args.classes
    )
    
    # Create processor
    print("Initializing OD + Segmentation Processor...")
    processor = OdSegmentationProcessor(
        args.od_modelpath,
        args.seg_modelpath,
        od_params
    )
    
    # Run based on input type
    if args.image:
        processor.run_image(args.image, args.loop)
    elif args.video:
        processor.run_video(args.video, loop=args.loop, fps_only=args.fps_only, 
                          target_fps=args.target_fps)
    elif args.rpicam or args.usbcam:
        processor.run_video(use_rpicam=args.rpicam, use_usbcam=args.usbcam,
                          camera_index=args.camera_index, fps_only=args.fps_only)


if __name__ == "__main__":
    main()