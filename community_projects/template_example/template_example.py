#!/usr/bin/env python3
"""
DEEPX Community Project Template
A template for creating new DEEPX NPU projects on Raspberry Pi 5

This template demonstrates:
- DEEPX NPU integration
- Configuration management
- Camera input handling
- Error handling and logging
- Performance monitoring
- Extensible architecture

Author: sixfab Community
"""

import os
import sys
import argparse
import logging
import time
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

class ConfigManager:
    """Configuration management for DEEPX projects"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                logging.info(f"Configuration loaded from {self.config_path}")
                return config
            else:
                logging.warning(f"Config file {self.config_path} not found, using defaults")
                return self.get_default_config()
        except Exception as e:
            logging.error(f"Error loading config: {e}")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'project': {
                'name': 'Template Example',
                'version': '1.0.0'
            },
            'camera': {
                'resolution': [640, 480],
                'fps': 30,
                'auto_exposure': True
            },
            'processing': {
                'confidence_threshold': 0.5,
                'nms_threshold': 0.4,
                'max_detections': 100
            },
            'display': {
                'show_fps': True,
                'show_confidence': True,
                'font_scale': 0.7
            },
            'output': {
                'save_results': False,
                'result_format': 'json'
            },
            'deepx': {
                'model_path': 'auto',
                'device_id': 0,
                'batch_size': 1
            },
            'logging': {
                'level': 'INFO',
                'file': 'template_example.log'
            }
        }
    
    def get_value(self, key_path: str, default=None) -> Any:
        """Get configuration value using dot notation"""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def save_config(self) -> None:
        """Save current configuration to file"""
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            logging.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logging.error(f"Error saving config: {e}")

class PerformanceMonitor:
    """Monitor performance metrics"""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.frame_times = []
        self.inference_times = []
        
    def update(self, frame_time: float, inference_time: float = 0.0):
        """Update performance metrics"""
        self.frame_times.append(frame_time)
        if inference_time > 0:
            self.inference_times.append(inference_time)
        
        # Keep only recent measurements
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
        if len(self.inference_times) > self.window_size:
            self.inference_times.pop(0)
    
    def get_fps(self) -> float:
        """Calculate current FPS"""
        if not self.frame_times:
            return 0.0
        return 1.0 / np.mean(self.frame_times)
    
    def get_inference_fps(self) -> float:
        """Calculate inference FPS"""
        if not self.inference_times:
            return 0.0
        return 1.0 / np.mean(self.inference_times)

class TemplateProcessor:
    """Main processing class for DEEPX template example"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.model = None
        self.performance_monitor = PerformanceMonitor()
        
        # Processing parameters
        self.confidence_threshold = self.config.get_value('processing.confidence_threshold', 0.5)
        self.nms_threshold = self.config.get_value('processing.nms_threshold', 0.4)
        self.max_detections = self.config.get_value('processing.max_detections', 100)
        
        # Initialize model
        model_path = self.config.get_value('deepx.model_path', 'auto')
        self.load_model(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """Load DEEPX NPU model"""
        try:
            if model_path == 'auto':
                # Auto-detect model
                model_dirs = ['models', '../models', '../../models']
                for model_dir in model_dirs:
                    if os.path.exists(model_dir):
                        model_files = [f for f in os.listdir(model_dir) if f.endswith('.dxnn')]
                        if model_files:
                            model_path = os.path.join(model_dir, model_files[0])
                            break
            
            logging.info(f"Loading DEEPX model: {model_path}")
            
            # TODO: Replace with actual DEEPX model loading
            # import dx_rt
            # self.model = dx_rt.load_model(model_path)
            
            if os.path.exists(model_path):
                self.model = f"deepx_model:{model_path}"
                logging.info("✅ DEEPX model loaded successfully")
                return True
            else:
                logging.warning(f"Model file not found: {model_path}")
                logging.info("Using CPU fallback mode")
                self.model = "cpu_fallback"
                return False
                
        except Exception as e:
            logging.error(f"Error loading DEEPX model: {e}")
            self.model = "cpu_fallback"
            return False
    
    def run_inference(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run DEEPX NPU inference"""
        start_time = time.time()
        
        try:
            # Preprocess frame
            input_tensor = self.preprocess_frame(frame)
            
            if "deepx_model" in str(self.model):
                # TODO: Replace with actual DEEPX inference
                # outputs = self.model.infer(input_tensor)
                
                # Simulate inference for template
                time.sleep(0.01)  # Simulate inference time
                results = self.simulate_detections(frame.shape)
                
            else:
                # CPU fallback mode
                results = self.simulate_detections(frame.shape)
            
            inference_time = time.time() - start_time
            self.performance_monitor.update(0, inference_time)
            
            return results
            
        except Exception as e:
            logging.error(f"Inference error: {e}")
            return []
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for DEEPX inference"""
        # Standard preprocessing for object detection
        target_size = (640, 640)
        
        # Resize while maintaining aspect ratio
        h, w = frame.shape[:2]
        scale = min(target_size[0] / w, target_size[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(frame, (new_w, new_h))
        
        # Create padded image
        padded = np.full((*target_size, 3), 128, dtype=np.uint8)
        y_offset = (target_size[1] - new_h) // 2
        x_offset = (target_size[0] - new_w) // 2
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        # Convert to float and normalize
        normalized = padded.astype(np.float32) / 255.0
        
        return normalized
    
    def simulate_detections(self, frame_shape: Tuple[int, int, int]) -> List[Dict[str, Any]]:
        """Simulate detection results for template demonstration"""
        h, w = frame_shape[:2]
        
        # Create some dummy detections
        detections = []
        
        if np.random.random() > 0.3:  # 70% chance of detection
            num_detections = np.random.randint(1, 4)
            
            for i in range(num_detections):
                # Random bounding box
                x1 = np.random.randint(0, w // 2)
                y1 = np.random.randint(0, h // 2)
                x2 = np.random.randint(x1 + 50, min(x1 + 200, w))
                y2 = np.random.randint(y1 + 50, min(y1 + 200, h))
                
                # Random class and confidence
                classes = ['person', 'car', 'bicycle', 'dog', 'cat']
                class_name = np.random.choice(classes)
                confidence = np.random.uniform(0.5, 0.95)
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': confidence,
                    'class_name': class_name,
                    'class_id': classes.index(class_name)
                })
        
        return detections
    
    def draw_results(self, frame: np.ndarray, results: List[Dict[str, Any]]) -> np.ndarray:
        """Draw detection results on frame"""
        result_frame = frame.copy()
        
        # Colors for different classes
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), 
            (255, 255, 0), (255, 0, 255), (0, 255, 255)
        ]
        
        for detection in results:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            class_id = detection.get('class_id', 0)
            
            x1, y1, x2, y2 = map(int, bbox)
            color = colors[class_id % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            if self.config.get_value('display.show_confidence', True):
                label = f"{class_name}: {confidence:.2f}"
            else:
                label = class_name
            
            # Label background
            font_scale = self.config.get_value('display.font_scale', 0.7)
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
            )
            
            cv2.rectangle(
                result_frame,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Label text
            cv2.putText(
                result_frame,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                2
            )
        
        return result_frame
    
    def draw_fps_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw FPS overlay on frame"""
        if not self.config.get_value('display.show_fps', True):
            return frame
        
        fps = self.performance_monitor.get_fps()
        inference_fps = self.performance_monitor.get_inference_fps()
        
        # FPS text
        fps_text = f"FPS: {fps:.1f}"
        if inference_fps > 0:
            fps_text += f" | Inference: {inference_fps:.1f}"
        
        # Draw FPS background
        (text_width, text_height), baseline = cv2.getTextSize(
            fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        
        cv2.rectangle(
            frame,
            (10, 10),
            (20 + text_width, 20 + text_height + baseline),
            (0, 0, 0),
            -1
        )
        
        # Draw FPS text
        cv2.putText(
            frame,
            fps_text,
            (15, 15 + text_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        return frame

def setup_camera(source: str, resolution: Tuple[int, int], fps: int) -> Optional[cv2.VideoCapture]:
    """Setup camera capture"""
    try:
        if source == "auto":
            # Auto-detect camera
            for i in range(5):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    logging.info(f"Auto-detected camera: /dev/video{i}")
                    break
            else:
                logging.error("No camera detected")
                return None
                
        elif source == "rpi":
            cap = cv2.VideoCapture(0)
            logging.info("Using Raspberry Pi camera")
            
        elif source == "usb":
            for i in range(1, 5):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    logging.info(f"Using USB camera: /dev/video{i}")
                    break
            else:
                logging.error("No USB camera found")
                return None
                
        elif source.startswith("/dev/video"):
            device_id = int(source.split("video")[1])
            cap = cv2.VideoCapture(device_id)
            logging.info(f"Using camera device: {source}")
            
        elif os.path.isfile(source):
            cap = cv2.VideoCapture(source)
            logging.info(f"Using video file: {source}")
            
        else:
            logging.error(f"Invalid input source: {source}")
            return None
        
        if not cap.isOpened():
            logging.error(f"Failed to open camera: {source}")
            return None
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        cap.set(cv2.CAP_PROP_FPS, fps)
        
        # Verify settings
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
        logging.info(f"Camera configured: {actual_width}x{actual_height} @ {actual_fps}fps")
        
        return cap
        
    except Exception as e:
        logging.error(f"Error setting up camera: {e}")
        return None

def create_output_directory(output_path: str) -> str:
    """Create output directory if it doesn't exist"""
    try:
        os.makedirs(output_path, exist_ok=True)
        return output_path
    except Exception as e:
        logging.error(f"Error creating output directory: {e}")
        return "."

def setup_logging(log_level: str, log_file: str, verbose: bool = False):
    """Setup logging configuration"""
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(levelname)-8s | %(message)s'
    )
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level if verbose else logging.INFO)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    try:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    except Exception as e:
        logging.warning(f"Could not setup file logging: {e}")

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description="DEEPX Community Project Template",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Auto-detect camera
  %(prog)s --input rpi                  # Raspberry Pi camera
  %(prog)s --input /dev/video0          # Specific camera
  %(prog)s --input video.mp4            # Video file
  %(prog)s --config custom.yaml         # Custom config
  %(prog)s --debug --verbose            # Debug mode
        """
    )
    
    parser.add_argument(
        "--input",
        default="auto",
        help="Input source (auto/rpi/usb/device/file)"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--output",
        default="output/",
        help="Output directory"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without display (headless mode)"
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save processed frames"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_manager = ConfigManager(args.config)
    
    # Setup logging
    log_level = "DEBUG" if args.debug else config_manager.get_value('logging.level', 'INFO')
    log_file = config_manager.get_value('logging.file', 'template_example.log')
    setup_logging(log_level, log_file, args.verbose)
    
    # Project info
    project_name = config_manager.get_value('project.name', 'Template Example')
    project_version = config_manager.get_value('project.version', '1.0.0')
    
    logging.info("=" * 60)
    logging.info(f"🚀 Starting {project_name} v{project_version}")
    logging.info("   DEEPX Community Project Template")
    logging.info("=" * 60)
    
    # Create output directory
    output_dir = create_output_directory(args.output)
    
    # Initialize processor
    logging.info("Initializing DEEPX processor...")
    processor = TemplateProcessor(config_manager)
    
    # Setup camera
    resolution = config_manager.get_value('camera.resolution', [640, 480])
    fps = config_manager.get_value('camera.fps', 30)
    
    logging.info("Setting up camera...")
    cap = setup_camera(args.input, tuple(resolution), fps)
    if cap is None:
        logging.error("❌ Failed to setup camera")
        return 1
    
    # Main processing loop
    logging.info("Starting main processing loop...")
    logging.info("Controls: 'q' to quit, 'space' to pause, 's' to save frame")
    
    frame_count = 0
    paused = False
    
    try:
        while True:
            loop_start = time.time()
            
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    logging.warning("Failed to read frame")
                    break
                
                # Process frame
                results = processor.run_inference(frame)
                
                # Draw results
                result_frame = processor.draw_results(frame, results)
                result_frame = processor.draw_fps_overlay(result_frame)
                
                # Save frame if requested
                if args.save_frames and frame_count % 30 == 0:  # Save every 30th frame
                    frame_filename = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
                    cv2.imwrite(frame_filename, result_frame)
                
                frame_count += 1
                
                # Log detection info periodically
                if frame_count % 100 == 0:
                    fps = processor.performance_monitor.get_fps()
                    logging.info(f"Frame {frame_count}: {len(results)} detections, {fps:.1f} FPS")
            
            # Display frame (if not headless)
            if not args.no_display:
                window_name = f"{project_name} - Press 'q' to quit"
                cv2.imshow(window_name, result_frame if not paused else frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logging.info("Quit requested by user")
                    break
                elif key == ord(' '):
                    paused = not paused
                    logging.info(f"{'⏸️  Paused' if paused else '▶️  Resumed'}")
                elif key == ord('s'):
                    # Save current frame
                    save_path = os.path.join(output_dir, f"manual_save_{int(time.time())}.jpg")
                    cv2.imwrite(save_path, result_frame)
                    logging.info(f"Frame saved: {save_path}")
            
            # Update performance monitoring
            frame_time = time.time() - loop_start
            processor.performance_monitor.update(frame_time)
            
    except KeyboardInterrupt:
        logging.info("🛑 Interrupted by user (Ctrl+C)")
    
    except Exception as e:
        logging.error(f"❌ Unexpected error: {e}")
        return 1
    
    finally:
        # Cleanup
        if cap:
            cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        
        # Final statistics
        final_fps = processor.performance_monitor.get_fps()
        logging.info("=" * 60)
        logging.info("📊 Final Statistics:")
        logging.info(f"   Frames processed: {frame_count}")
        logging.info(f"   Average FPS: {final_fps:.1f}")
        logging.info(f"   Output directory: {output_dir}")
        logging.info("✅ Template example completed successfully!")
        logging.info("=" * 60)
    
    return 0

if __name__ == "__main__":
    exit(main())