import os
import cv2
import numpy as np
import json
import argparse
from dx_engine import InferenceEngine
from dx_engine import Configuration
from packaging import version
import time
import sys

# Try to import picamera2 for Raspberry Pi camera support
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    print("Warning: picamera2 not available. Raspberry Pi camera support disabled.")

class SegmentationParam:
    def __init__(self, id, name, r, g, b):
        self.id = id
        self.name = name
        self.color = (b, g, r)  # OpenCV uses BGR format

# 19 classes for cityscapes datasets
segmentation_config_19classes = [
    SegmentationParam(0, "road", 128, 64, 128),
    SegmentationParam(1, "sidewalk", 244, 35, 232),
    SegmentationParam(2, "building", 70, 70, 70),
    SegmentationParam(3, "wall", 102, 102, 156),
    SegmentationParam(4, "fence", 190, 153, 153),
    SegmentationParam(5, "pole", 153, 153, 153),
    SegmentationParam(6, "traffic light", 51, 255, 255),
    SegmentationParam(7, "traffic sign", 220, 220, 0),
    SegmentationParam(8, "vegetation", 107, 142, 35),
    SegmentationParam(9, "terrain", 152, 251, 152),
    SegmentationParam(10, "sky", 255, 0, 0),
    SegmentationParam(11, "person", 0, 51, 255),
    SegmentationParam(12, "rider", 255, 0, 0),
    SegmentationParam(13, "car", 255, 51, 0),
    SegmentationParam(14, "truck", 255, 51, 0),
    SegmentationParam(15, "bus", 255, 51, 0),
    SegmentationParam(16, "train", 0, 80, 100),
    SegmentationParam(17, "motorcycle", 0, 0, 230),
    SegmentationParam(18, "bicycle", 119, 11, 32)
]

segmentation_config_3classes = [
    SegmentationParam(0, "background", 0, 0, 0),
    SegmentationParam(1, "foot", 0, 128, 0),
    SegmentationParam(2, "body", 0, 0, 128),
]

def setup_camera(use_rpicam=False, camera_index=0, width=640, height=480, fps=30):
    """
    Setup camera based on the type requested.
    
    Args:
        use_rpicam: If True, use Raspberry Pi camera, otherwise use USB camera
        camera_index: Index of the USB camera (default: 0)
        width: Camera width resolution
        height: Camera height resolution
        fps: Frames per second
    
    Returns:
        Camera object and a function to read frames
    """
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
            # Convert RGB to BGR for OpenCV
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

def preprocess(frame, input_size):
    """
    Preprocess the input frame for segmentation model
    
    Args:
        frame: Input frame (BGR format)
        input_size: Tuple of (height, width) for model input
    
    Returns:
        Preprocessed frame
    """
    resized = cv2.resize(frame, (input_size[1], input_size[0]))
    # Convert BGR to RGB if needed by model
    # resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return resized

def apply_segmentation_mask(output_data, seg_config, num_classes, model_output_shape):
    """
    Apply color map to segmentation output
    
    Args:
        output_data: Model output (can be uint16 or float)
        seg_config: List of SegmentationParam objects
        num_classes: Number of classes
        model_output_shape: Original shape from model output
    
    Returns:
        Colored segmentation mask (BGR format)
    """
    print(f"Output data shape: {output_data.shape}, dtype: {output_data.dtype}")
    
    # Handle different output shapes
    if len(output_data.shape) == 4:
        # Shape: (batch, num_classes, height, width) or (batch, height, width, num_classes)
        output_data = np.squeeze(output_data, axis=0)  # Remove batch dimension
    
    if len(output_data.shape) == 3:
        # Determine if shape is (C, H, W) or (H, W, C)
        if output_data.shape[0] == num_classes:
            # Shape: (num_classes, height, width)
            height, width = output_data.shape[1], output_data.shape[2]
            class_map = np.argmax(output_data, axis=0)
        elif output_data.shape[2] == num_classes:
            # Shape: (height, width, num_classes)
            height, width = output_data.shape[0], output_data.shape[1]
            class_map = np.argmax(output_data, axis=2)
        else:
            raise ValueError(f"Unexpected output shape: {output_data.shape}")
    elif len(output_data.shape) == 2:
        # Already a 2D class map
        height, width = output_data.shape
        class_map = output_data.astype(np.int32)
    elif len(output_data.shape) == 1:
        # Flattened output - need to reshape
        # Try to infer dimensions
        total_pixels = output_data.shape[0]
        if total_pixels % num_classes == 0:
            # It's probably flattened class probabilities
            pixels = total_pixels // num_classes
            height = width = int(np.sqrt(pixels))
            output_data = output_data.reshape(num_classes, height, width)
            class_map = np.argmax(output_data, axis=0)
        else:
            # It's probably flattened class indices
            height = width = int(np.sqrt(total_pixels))
            class_map = output_data.reshape(height, width).astype(np.int32)
    else:
        raise ValueError(f"Unsupported output shape: {output_data.shape}")
    
    print(f"Class map shape: {class_map.shape}, unique classes: {np.unique(class_map)}")
    
    # Create colored segmentation result
    seg_result = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Apply color map
    for seg_class in seg_config:
        mask = (class_map == seg_class.id)
        seg_result[mask] = seg_class.color
    
    return seg_result

class SegmentationModel:
    def __init__(self, model_path, parameter=0):
        """
        Initialize segmentation model
        
        Args:
            model_path: Path to the model file
            parameter: 0 for 19 classes (cityscapes), 1 for 3 classes
        """
        self.ie = InferenceEngine(model_path)
        
        # Check version
        if version.parse(Configuration().get_version()) < version.parse("3.0.0"):
            print("DX-RT version 3.0.0 or higher is required. Please update DX-RT to the latest version.")
            exit()
        
        if version.parse(self.ie.get_model_version()) < version.parse('7'):
            print("dxnn files format version 7 or higher is required. Please update/re-export the model.")
            exit()
        
        # Get input shape
        input_info = self.ie.get_input_tensors_info()[0]
        self.input_height = input_info['shape'][1]
        self.input_width = input_info['shape'][2]
        
        # Get output info
        output_info = self.ie.get_output_tensors_info()[0]
        self.output_dtype = output_info['dtype']
        self.output_shape = output_info['shape']
        
        # Set segmentation configuration
        if parameter == 0:
            self.seg_config = segmentation_config_19classes
            self.num_classes = 19
        else:
            self.seg_config = segmentation_config_3classes
            self.num_classes = 3
        
        print(f"Model input size: {self.input_width}x{self.input_height}")
        print(f"Model output shape: {self.output_shape}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Output dtype: {self.output_dtype}")
    
    def run(self, frame):
        """
        Run segmentation on input frame
        
        Args:
            frame: Input frame (BGR format)
        
        Returns:
            Segmentation mask (colored)
        """
        # Preprocess
        input_frame = preprocess(frame, (self.input_height, self.input_width))
        
        # Run inference
        outputs = self.ie.run([input_frame])
        
        # Post-process
        output_data = outputs[0]
        seg_mask = apply_segmentation_mask(
            output_data, 
            self.seg_config, 
            self.num_classes,
            self.output_shape
        )
        
        return seg_mask

def run_segmentation(model_path, parameter, input_source=None, use_rpicam=False, 
                     use_usbcam=False, camera_index=0, fps_only=False, 
                     loop=False, target_fps=0):
    """
    Run segmentation inference
    
    Args:
        model_path: Path to model file
        parameter: 0 for 19 classes, 1 for 3 classes
        input_source: Path to image or video file (None for camera)
        use_rpicam: Use Raspberry Pi camera
        use_usbcam: Use USB camera
        camera_index: USB camera index
        fps_only: Only show FPS, don't visualize
        loop: Loop video file
        target_fps: Target FPS for video playback
    """
    # Initialize model
    seg_model = SegmentationModel(model_path, parameter)
    
    # Determine input mode
    is_image = False
    is_video = False
    is_camera = use_rpicam or use_usbcam
    
    if input_source and os.path.exists(input_source):
        ext = os.path.splitext(input_source)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            is_image = True
        elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
            is_video = True
    
    # Process based on input type
    if is_image:
        print(f"Processing image: {input_source}")
        frame = cv2.imread(input_source)
        if frame is None:
            print(f"Error: Could not read image {input_source}")
            return
        
        # Run inference
        start_time = time.time()
        seg_mask = seg_model.run(frame)
        end_time = time.time()
        
        # Resize mask to match original frame size
        seg_mask_resized = cv2.resize(seg_mask, (frame.shape[1], frame.shape[0]))
        
        # Overlay on original frame
        result = cv2.addWeighted(frame, 0.6, seg_mask_resized, 0.4, 0)
        
        # Save result
        cv2.imwrite("result.jpg", result)
        print(f"Result saved to result.jpg")
        print(f"Inference time: {(end_time - start_time) * 1000:.2f} ms")
        
        if not fps_only:
            cv2.imshow("Segmentation Result", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    elif is_video or is_camera:
        # Setup video capture or camera
        if is_camera:
            camera, read_frame = setup_camera(use_rpicam=use_rpicam, camera_index=camera_index)
        else:
            print(f"Processing video: {input_source}")
            cap = cv2.VideoCapture(input_source)
            if not cap.isOpened():
                print(f"Error: Could not open video {input_source}")
                return
            read_frame = cap.read
            
            # Get video info
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"Video FPS: {video_fps}")
            print(f"Total frames: {total_frames}")
        
        # Setup display window
        if not fps_only:
            cv2.namedWindow("Segmentation Result", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Segmentation Result", 960, 540)
        
        print("Press 'q' to quit...")
        
        frame_count = 0
        total_time = 0
        
        # Flag to print debug info only once
        debug_printed = False
        
        while True:
            ret, frame = read_frame()
            
            if not ret:
                if is_video and loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print("\nVideo ended or failed to read frame.")
                    break
            
            # Run inference
            start_time = time.time()
            
            # Print debug info only for first frame
            if not debug_printed:
                print(f"Input frame shape: {frame.shape}")
                debug_printed = True
            
            seg_mask = seg_model.run(frame)
            end_time = time.time()
            
            inference_time = end_time - start_time
            total_time += inference_time
            frame_count += 1
            
            # Resize mask to match original frame size
            seg_mask_resized = cv2.resize(seg_mask, (frame.shape[1], frame.shape[0]))
            
            # Overlay on original frame
            result = cv2.addWeighted(frame, 0.6, seg_mask_resized, 0.4, 0)
            
            # Display FPS on frame
            fps = 1.0 / inference_time if inference_time > 0 else 0
            avg_fps = frame_count / total_time if total_time > 0 else 0
            
            if not fps_only:
                cv2.putText(result, f"FPS: {fps:.1f} | Avg: {avg_fps:.1f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 255, 0), 2)
                cv2.imshow("Segmentation Result", result)
            
            # Print FPS info
            if frame_count % 30 == 0 or fps_only:
                print(f"Frame {frame_count} | FPS: {fps:.1f} | Avg FPS: {avg_fps:.1f}")
            
            # Handle target FPS
            if target_fps > 0 and not is_camera:
                target_frame_time = 1.0 / target_fps
                if inference_time < target_frame_time:
                    time.sleep(target_frame_time - inference_time)
            
            # Check for quit
            if not fps_only:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # Cleanup
        if is_camera:
            if use_rpicam and PICAMERA2_AVAILABLE:
                camera.stop()
            else:
                camera.release()
        else:
            cap.release()
        
        cv2.destroyAllWindows()
        
        # Print statistics
        print("\n" + "="*50)
        print(f"Total frames processed: {frame_count}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average FPS: {frame_count / total_time if total_time > 0 else 0:.2f}")
        print(f"Average inference time: {(total_time / frame_count * 1000) if frame_count > 0 else 0:.2f} ms")
        print("="*50)
    
    else:
        print("Error: No valid input source specified")
        print("Use --image, --video, --rpicam, or --usbcam")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Segmentation Model Demo with Camera Support')
    parser.add_argument('-m', '--model', required=True, type=str,
                        help='Path to dxnn model file')
    parser.add_argument('-p', '--parameter', default=0, type=int,
                        help='Segmentation parameter: 0=19 classes (cityscapes), 1=3 classes')
    parser.add_argument('-i', '--image', type=str,
                        help='Path to input image file')
    parser.add_argument('-v', '--video', type=str,
                        help='Path to input video file')
    parser.add_argument('--rpicam', action='store_true',
                        help='Use Raspberry Pi camera')
    parser.add_argument('--usbcam', action='store_true',
                        help='Use USB camera')
    parser.add_argument('--camera_index', default=0, type=int,
                        help='USB camera index (default: 0)')
    parser.add_argument('--fps_only', action='store_true',
                        help='Only show FPS, do not visualize')
    parser.add_argument('-l', '--loop', action='store_true',
                        help='Loop video file')
    parser.add_argument('--target_fps', default=0, type=int,
                        help='Target FPS for video playback')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.rpicam and args.usbcam:
        print("Error: Cannot use both --rpicam and --usbcam at the same time")
        parser.print_help()
        exit(1)
    
    input_count = sum([bool(args.image), bool(args.video), args.rpicam, args.usbcam])
    if input_count == 0:
        print("Error: No input source specified")
        parser.print_help()
        exit(1)
    
    if input_count > 1:
        print("Error: Only one input source can be specified at a time")
        parser.print_help()
        exit(1)
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        exit(1)
    
    # Determine input source
    input_source = args.image if args.image else args.video
    
    print("Starting segmentation app, please wait...")
    run_segmentation(
        model_path=args.model,
        parameter=args.parameter,
        input_source=input_source,
        use_rpicam=args.rpicam,
        use_usbcam=args.usbcam,
        camera_index=args.camera_index,
        fps_only=args.fps_only,
        loop=args.loop,
        target_fps=args.target_fps
    )