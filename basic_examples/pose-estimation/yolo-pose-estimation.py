#!/usr/bin/env python3
"""
DEEPX Pose Estimation Example - Python Implementation

This script performs pose estimation using DEEPX NPU runtime, similar to the C++ version.
It supports image files, video files, and camera input with real-time visualization.

Requirements:
- dx_engine (DEEPX NPU runtime)
- OpenCV (cv2)
- NumPy
- argparse
"""

import os
import cv2
import numpy as np
import argparse
import time
import threading
import queue
from typing import List, Tuple, Optional, Dict, Any
import json

# Try to import DEEPX runtime
try:
    from dx_engine import InferenceEngine, InferenceOption
except ImportError:
    print("Error: dx_engine not found. Please install DEEPX NPU runtime.")
    exit(1)

# Try to import Picamera2 for RPi camera support
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False

class PoseEstimationConfig:
    """Configuration class for pose estimation parameters"""
    
    def __init__(self):
        # YOLOv5s6 pose estimation default parameters (similar to C++ yolov5s6_pose_640)
        self.name = "yolov5s6_pose_640"
        self.width = 640
        self.height = 640
        self.score_threshold = 0.3
        self.iou_threshold = 0.4
        self.num_classes = 1  # person class for pose estimation
        self.num_keypoints = 17  # COCO pose format (17 keypoints)
        self.input_format = "RGB"
        self.normalization_mean = [0.0, 0.0, 0.0]
        self.normalization_std = [255.0, 255.0, 255.0]
        
        # COCO pose keypoint connections for drawing skeleton
        self.pose_pairs = [
            (0, 1),   # nose -> left_eye
            (0, 2),   # nose -> right_eye
            (1, 3),   # left_eye -> left_ear
            (2, 4),   # right_eye -> right_ear
            (5, 6),   # left_shoulder -> right_shoulder
            (5, 7),   # left_shoulder -> left_elbow
            (7, 9),   # left_elbow -> left_wrist
            (6, 8),   # right_shoulder -> right_elbow
            (8, 10),  # right_elbow -> right_wrist
            (5, 11),  # left_shoulder -> left_hip
            (6, 12),  # right_shoulder -> right_hip
            (11, 12), # left_hip -> right_hip
            (11, 13), # left_hip -> left_knee
            (13, 15), # left_knee -> left_ankle
            (12, 14), # right_hip -> right_knee
            (14, 16)  # right_knee -> right_ankle
        ]
        
        # COCO keypoint names
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Colors for visualization (BGR format)
        self.skeleton_color = (0, 255, 0)  # Green for skeleton
        self.keypoint_color = (0, 0, 255)  # Red for keypoints
        self.bbox_color = (255, 0, 0)      # Blue for bounding boxes

class BoundingBox:
    """Bounding box class for pose detection results"""
    
    def __init__(self, x1: float, y1: float, x2: float, y2: float, 
                 confidence: float, class_id: int, keypoints: Optional[List[Tuple[float, float, float]]] = None):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.confidence = confidence
        self.class_id = class_id
        self.keypoints = keypoints or []  # List of (x, y, confidence) tuples

class PoseEstimator:
    """Main pose estimation class"""
    
    def __init__(self, model_path: str, config: PoseEstimationConfig):
        self.model_path = model_path
        self.config = config
        self.inference_engine = None
        self.frame_buffers = 5
        self.processed_count = 0
        self.results_queue = queue.Queue(maxsize=self.frame_buffers)
        self.display_thread = None
        self.stop_processing = False
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the DEEPX model"""
        try:
            # Create InferenceOption object properly
            inference_options = InferenceOption()
            inference_options.useORT = True
            inference_options.devices = [0]  # Use device 0
            
            # Create inference engine with DEEPX NPU
            self.inference_engine = InferenceEngine(self.model_path, inference_options)
            print(f"Model loaded successfully from: {self.model_path}")
            
            # Try to get model input/output information if available
            try:
                if hasattr(self.inference_engine, 'get_input_shapes'):
                    print(f"Model input shape: {self.inference_engine.get_input_shapes()}")
                if hasattr(self.inference_engine, 'get_output_shapes'):
                    print(f"Model output shape: {self.inference_engine.get_output_shapes()}")
            except:
                pass  # Skip if methods don't exist
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """
        Preprocess input image for pose estimation
        Returns: (processed_image, ratio, (pad_w, pad_h))
        """
        # Letterbox resize to maintain aspect ratio
        h, w = image.shape[:2]
        new_h, new_w = self.config.height, self.config.width
        
        # Calculate scaling ratio
        ratio = min(new_h / h, new_w / w)
        
        # Calculate new size after scaling
        new_unpad_h = int(round(h * ratio))
        new_unpad_w = int(round(w * ratio))
        
        # Calculate padding
        pad_w = (new_w - new_unpad_w) / 2
        pad_h = (new_h - new_unpad_h) / 2
        
        # Resize image
        if (h, w) != (new_unpad_h, new_unpad_w):
            image = cv2.resize(image, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)
        
        # Add padding
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        
        image = cv2.copyMakeBorder(image, top, bottom, left, right, 
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        # Convert BGR to RGB if needed
        if self.config.input_format == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize
        image = image.astype(np.float32)
        for i in range(3):
            image[:, :, i] = (image[:, :, i] - self.config.normalization_mean[i]) / self.config.normalization_std[i]
        
        # Add batch dimension and change to CHW format
        image = np.transpose(image, (2, 0, 1))  # HWC to CHW
        image = np.expand_dims(image, axis=0)   # Add batch dimension
        
        return image, ratio, (pad_w, pad_h)
    
    def postprocess_outputs(self, outputs: List[np.ndarray], 
                          original_shape: Tuple[int, int], 
                          ratio: float, 
                          pad: Tuple[float, float]) -> List[BoundingBox]:
        """
        Postprocess model outputs to extract bounding boxes and keypoints
        """
        detections = []
        
        if not outputs or len(outputs) == 0:
            return detections
        
        # Assume output format: [batch, (bbox + conf + keypoints), anchors]
        # For YOLO pose: [1, 56, 8400] where 56 = 4(bbox) + 1(conf) + 51(17*3 keypoints)
        output = outputs[0]  # Take first output
        
        if len(output.shape) == 3:
            output = output[0]  # Remove batch dimension
        
        # Transpose if needed: [anchors, features] format
        if output.shape[0] == 56 or output.shape[0] == 4 + 1 + self.config.num_keypoints * 3:
            output = output.T
        
        orig_h, orig_w = original_shape
        
        for detection in output:
            # Extract bbox, confidence, and keypoints
            if len(detection) < 5 + self.config.num_keypoints * 3:
                continue
                
            bbox = detection[:4]  # x_center, y_center, width, height
            confidence = detection[4]
            keypoint_data = detection[5:5 + self.config.num_keypoints * 3]
            
            if confidence < self.config.score_threshold:
                continue
            
            # Convert from center format to corner format
            x_center, y_center, width, height = bbox
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            
            # Scale back to original image coordinates
            x1 = (x1 - pad[0]) / ratio
            y1 = (y1 - pad[1]) / ratio
            x2 = (x2 - pad[0]) / ratio
            y2 = (y2 - pad[1]) / ratio
            
            # Clip to image bounds
            x1 = max(0, min(orig_w, x1))
            y1 = max(0, min(orig_h, y1))
            x2 = max(0, min(orig_w, x2))
            y2 = max(0, min(orig_h, y2))
            
            # Process keypoints
            keypoints = []
            for i in range(0, len(keypoint_data), 3):
                if i + 2 < len(keypoint_data):
                    kp_x = keypoint_data[i]
                    kp_y = keypoint_data[i + 1]
                    kp_conf = keypoint_data[i + 2]
                    
                    # Scale keypoints back to original coordinates
                    kp_x = (kp_x - pad[0]) / ratio
                    kp_y = (kp_y - pad[1]) / ratio
                    
                    # Clip keypoints to image bounds
                    kp_x = max(0, min(orig_w, kp_x))
                    kp_y = max(0, min(orig_h, kp_y))
                    
                    keypoints.append((kp_x, kp_y, kp_conf))
            
            bbox_obj = BoundingBox(x1, y1, x2, y2, confidence, 0, keypoints)
            detections.append(bbox_obj)
        
        # Apply NMS
        detections = self._apply_nms(detections)
        
        return detections
    
    def _apply_nms(self, detections: List[BoundingBox]) -> List[BoundingBox]:
        """Apply Non-Maximum Suppression"""
        if not detections:
            return []
        
        # Sort by confidence
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        keep = []
        while detections:
            current = detections.pop(0)
            keep.append(current)
            
            # Remove overlapping boxes
            detections = [det for det in detections 
                         if self._calculate_iou(current, det) <= self.config.iou_threshold]
        
        return keep
    
    def _calculate_iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """Calculate Intersection over Union"""
        # Calculate intersection
        x1 = max(box1.x1, box2.x1)
        y1 = max(box1.y1, box2.y1)
        x2 = min(box1.x2, box2.x2)
        y2 = min(box1.y2, box2.y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
        area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def draw_pose(self, image: np.ndarray, detections: List[BoundingBox]) -> np.ndarray:
        """Draw pose estimation results on image"""
        result_image = image.copy()
        
        for detection in detections:
            # Draw bounding box
            cv2.rectangle(result_image, 
                         (int(detection.x1), int(detection.y1)), 
                         (int(detection.x2), int(detection.y2)), 
                         self.config.bbox_color, 2)
            
            # Draw confidence score
            conf_text = f"Person: {detection.confidence:.2f}"
            cv2.putText(result_image, conf_text, 
                       (int(detection.x1), int(detection.y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.config.bbox_color, 2)
            
            # Draw keypoints
            for i, (x, y, conf) in enumerate(detection.keypoints):
                if conf > 0.3:  # Keypoint confidence threshold
                    cv2.circle(result_image, (int(x), int(y)), 3, self.config.keypoint_color, -1)
                    
                    # Optionally draw keypoint names (commented out for cleaner display)
                    # if i < len(self.config.keypoint_names):
                    #     cv2.putText(result_image, self.config.keypoint_names[i], 
                    #                (int(x) + 5, int(y) - 5),
                    #                cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.config.keypoint_color, 1)
            
            # Draw skeleton connections
            for start_idx, end_idx in self.config.pose_pairs:
                if (start_idx < len(detection.keypoints) and 
                    end_idx < len(detection.keypoints)):
                    
                    start_point = detection.keypoints[start_idx]
                    end_point = detection.keypoints[end_idx]
                    
                    # Check if both keypoints are confident enough
                    if start_point[2] > 0.3 and end_point[2] > 0.3:
                        cv2.line(result_image,
                                (int(start_point[0]), int(start_point[1])),
                                (int(end_point[0]), int(end_point[1])),
                                self.config.skeleton_color, 2)
        
        return result_image
    
    def process_image(self, image_path: str, save_result: bool = True) -> None:
        """Process a single image"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return
        
        print(f"Processing image: {image_path}")
        start_time = time.time()
        
        # Preprocess
        processed_image, ratio, pad = self.preprocess_image(image)
        
        # Run inference
        outputs = self.inference_engine.run(processed_image)
        
        # Postprocess
        detections = self.postprocess_outputs(outputs, image.shape[:2], ratio, pad)
        
        # Draw results
        result_image = self.draw_pose(image, detections)
        
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000
        
        print(f"Inference time: {inference_time:.2f} ms")
        print(f"Detected {len(detections)} person(s)")
        
        # Display results
        cv2.imshow("Pose Estimation Result", result_image)
        
        if save_result:
            result_path = "pose_estimation_result.jpg"
            cv2.imwrite(result_path, result_image)
            print(f"Result saved to: {result_path}")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def process_video(self, video_path: str, loop: bool = False) -> None:
        """Process video file"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        print(f"Processing video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Video FPS: {fps}")
        
        frame_count = 0
        total_time = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                if loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print("Video processing completed.")
                    break
            
            start_time = time.time()
            
            # Preprocess
            processed_frame, ratio, pad = self.preprocess_image(frame)
            
            # Run inference
            outputs = self.inference_engine.run(processed_frame)
            
            # Postprocess
            detections = self.postprocess_outputs(outputs, frame.shape[:2], ratio, pad)
            
            # Draw results
            result_frame = self.draw_pose(frame, detections)
            
            end_time = time.time()
            frame_time = (end_time - start_time) * 1000
            total_time += frame_time
            frame_count += 1
            
            # Display FPS
            avg_fps = 1000 / (total_time / frame_count) if frame_count > 0 else 0
            cv2.putText(result_frame, f"FPS: {avg_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Pose Estimation Video", result_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"Total frames processed: {frame_count}")
        print(f"Average processing time: {total_time / frame_count:.2f} ms per frame")
        print(f"Average FPS: {1000 * frame_count / total_time:.2f}")
    
    def process_camera_usb(self, camera_id: int = 0) -> None:
        """Process USB camera input"""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Could not open USB camera {camera_id}")
            return
        
        print(f"Processing USB camera input: {camera_id}")
        
        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Camera resolution: {width}x{height}")
        print(f"Camera FPS: {fps}")
        print("Press 'q' or ESC to quit")
        
        frame_count = 0
        total_time = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera")
                break
            
            start_time = time.time()
            
            # Preprocess
            processed_frame, ratio, pad = self.preprocess_image(frame)
            
            # Run inference
            outputs = self.inference_engine.run(processed_frame)
            
            # Postprocess
            detections = self.postprocess_outputs(outputs, frame.shape[:2], ratio, pad)
            
            # Draw results
            result_frame = self.draw_pose(frame, detections)
            
            end_time = time.time()
            frame_time = (end_time - start_time) * 1000
            total_time += frame_time
            frame_count += 1
            
            # Display FPS and detection info
            avg_fps = 1000 / (total_time / frame_count) if frame_count > 0 else 0
            cv2.putText(result_frame, f"FPS: {avg_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(result_frame, f"Detected: {len(detections)} person(s)", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Pose Estimation Camera", result_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"Total frames processed: {frame_count}")
        if frame_count > 0:
            print(f"Average processing time: {total_time / frame_count:.2f} ms per frame")
            print(f"Average FPS: {1000 * frame_count / total_time:.2f}")

    def process_camera_rpi(self) -> None:
        """Process Raspberry Pi camera input"""
        if not PICAMERA_AVAILABLE:
            print("Error: Picamera2 not available. Please install picamera2.")
            return

        # Initialize Picamera2
        picam2 = Picamera2()
        
        # Configure camera
        config = picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        picam2.configure(config)
        
        # Start camera
        picam2.start()
        print("RPi Camera started")
        print("Press 'q' or ESC to quit")

        try:
            frame_count = 0
            total_time = 0

            while True:
                # Capture frame
                frame = picam2.capture_array()
                
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                start_time = time.time()
                
                # Preprocess
                processed_frame, ratio, pad = self.preprocess_image(frame_bgr)
                
                # Run inference
                outputs = self.inference_engine.run(processed_frame)
                
                # Postprocess
                detections = self.postprocess_outputs(outputs, frame_bgr.shape[:2], ratio, pad)
                
                # Draw results
                result_frame = self.draw_pose(frame_bgr, detections)
                
                end_time = time.time()
                frame_time = (end_time - start_time) * 1000
                total_time += frame_time
                frame_count += 1
                
                # Display FPS and detection info
                avg_fps = 1000 / (total_time / frame_count) if frame_count > 0 else 0
                cv2.putText(result_frame, f"FPS: {avg_fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(result_frame, f"Detected: {len(detections)} person(s)", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(result_frame, f"RPi Camera", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow("Pose Estimation RPi Camera", result_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break

        finally:
            picam2.stop()
            cv2.destroyAllWindows()
            
            print(f"Total frames processed: {frame_count}")
            if frame_count > 0:
                print(f"Average processing time: {total_time / frame_count:.2f} ms per frame")
                print(f"Average FPS: {1000 * frame_count / total_time:.2f}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="DEEPX Pose Estimation - Python Implementation")
    parser.add_argument("-m", "--model", required=True, 
                       help="Path to the DEEPX pose estimation model (.dxnn file)")
    parser.add_argument("-i", "--image", 
                       help="Path to input image file")
    parser.add_argument("-v", "--video", 
                       help="Path to input video file")
    parser.add_argument("--usbcam", action="store_true",
                       help="Use USB camera input")
    parser.add_argument("--rpicam", action="store_true",
                       help="Use Raspberry Pi camera input")
    parser.add_argument("--camera_id", type=int, default=0,
                       help="USB camera ID (default: 0)")
    parser.add_argument("-l", "--loop", action="store_true",
                       help="Loop video playback")
    parser.add_argument("--config", 
                       help="Path to configuration JSON file")
    parser.add_argument("--score_threshold", type=float, default=0.3,
                       help="Detection confidence threshold (default: 0.3)")
    parser.add_argument("--iou_threshold", type=float, default=0.4,
                       help="NMS IoU threshold (default: 0.4)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    input_count = sum([bool(args.image), bool(args.video), args.usbcam, args.rpicam])
    if input_count != 1:
        print("Error: Please specify exactly one input source (image, video, usbcam, or rpicam)")
        return
    
    if args.image and not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return
    
    if args.video and not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return
    
    # Load configuration
    config = PoseEstimationConfig()
    if args.config and os.path.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                config_data = json.load(f)
            # Update config with loaded values
            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")
    
    # Override with command line arguments
    config.score_threshold = args.score_threshold
    config.iou_threshold = args.iou_threshold
    
    print(f"DEEPX Pose Estimation - Python Implementation")
    print(f"Model: {args.model}")
    print(f"Configuration: {config.name}")
    print(f"Input size: {config.width}x{config.height}")
    print(f"Score threshold: {config.score_threshold}")
    print(f"IoU threshold: {config.iou_threshold}")
    print("-" * 50)
    
    try:
        # Create pose estimator
        pose_estimator = PoseEstimator(args.model, config)
        
        # Process input
        if args.image:
            pose_estimator.process_image(args.image)
        elif args.video:
            pose_estimator.process_video(args.video, args.loop)
        elif args.usbcam:
            pose_estimator.process_camera_usb(args.camera_id)
        elif args.rpicam:
            pose_estimator.process_camera_rpi()
            
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()