import os
import cv2
import numpy as np
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

INPUT_CAPTURE_PERIOD_MS = 1
DISPLAY_WINDOW_NAME = "DENOISE"

WIDTH = 512
HEIGHT = 512

CAMERA_FRAME_WIDTH = 800
CAMERA_FRAME_HEIGHT = 600

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
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        if not cap.isOpened():
            print(f"Error: Cannot open USB camera with index {camera_index}")
            sys.exit(1)
        
        def read_frame():
            return cap.read()
        
        return cap, read_frame

def get_noise_image(src, mean=0.0, std=15.0):
    """
    Add Gaussian noise to image
    
    Args:
        src: Source image (BGR format)
        mean: Mean of Gaussian noise
        std: Standard deviation of Gaussian noise
    
    Returns:
        Noisy image
    """
    # Create Gaussian noise
    gaussian_noise = np.random.normal(mean, std, src.shape).astype(np.int16)
    
    # Convert source to int16 for addition
    add_weight = src.astype(np.int16)
    
    # Add noise
    noisy = add_weight + gaussian_noise
    
    # Clip values to valid range and convert back to uint8
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    return noisy

class DenoiseModel:
    def __init__(self, model_path):
        """
        Initialize denoise model
        
        Args:
            model_path: Path to the model file
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
        self.input_channels = input_info['shape'][3] if len(input_info['shape']) > 3 else 3
        
        # Get output info
        output_info = self.ie.get_output_tensors_info()[0]
        self.output_shape = output_info['shape']
        
        # Determine if output is grayscale or color
        # Shape can be [1, 1, H, W] for grayscale or [1, 3, H, W] / [1, H, W, 3] for color
        if len(self.output_shape) == 4:
            if self.output_shape[1] == 1:
                self.is_grayscale = True
                self.output_height = self.output_shape[2]
                self.output_width = self.output_shape[3]
            elif self.output_shape[1] == 3:
                self.is_grayscale = False
                self.output_height = self.output_shape[2]
                self.output_width = self.output_shape[3]
            elif self.output_shape[3] == 3:
                self.is_grayscale = False
                self.output_height = self.output_shape[1]
                self.output_width = self.output_shape[2]
            else:
                self.is_grayscale = True
                self.output_height = self.output_shape[2]
                self.output_width = self.output_shape[3]
        else:
            self.is_grayscale = False
            self.output_height = self.output_shape[0]
            self.output_width = self.output_shape[1]
        
        print(f"Model input size: {self.input_width}x{self.input_height}")
        print(f"Model output shape: {self.output_shape}")
        print(f"Grayscale output: {self.is_grayscale}")
    
    def run(self, frame):
        """
        Run denoising on input frame
        
        Args:
            frame: Input frame (BGR format, should be resized to model input size)
        
        Returns:
            Denoised frame (BGR format)
        """
        # Run inference
        outputs = self.ie.run([frame])
        
        # Post-process
        output_data = outputs[0]
        
        # Remove batch dimension if present
        if len(output_data.shape) == 4:
            output_data = np.squeeze(output_data, axis=0)
        
        # Convert output to image based on format
        if self.is_grayscale:
            # Output is grayscale: [1, H, W] or [H, W, 1] or [H, W]
            if len(output_data.shape) == 3:
                if output_data.shape[0] == 1:
                    # Shape: [1, H, W]
                    output_data = output_data[0]
                elif output_data.shape[2] == 1:
                    # Shape: [H, W, 1]
                    output_data = output_data[:, :, 0]
            
            # Normalize to 0-255 range
            output_gray = (output_data * 255.0).clip(0, 255).astype(np.uint8)
            
            # Convert grayscale to BGR
            output_frame = cv2.cvtColor(output_gray, cv2.COLOR_GRAY2BGR)
        else:
            # Output is color
            if output_data.shape[0] == 3:
                # Shape: [3, H, W] - need to transpose to [H, W, 3]
                output_data = np.transpose(output_data, (1, 2, 0))
            
            # Normalize to 0-255 range
            output_frame = (output_data * 255.0).clip(0, 255).astype(np.uint8)
        
        return output_frame

def run_denoise(model_path, input_source=None, use_rpicam=False, use_usbcam=False, 
                camera_index=0, mean=0.0, std=15.0):
    """
    Run denoising inference
    
    Args:
        model_path: Path to model file
        input_source: Path to image or video file (None for camera)
        use_rpicam: Use Raspberry Pi camera
        use_usbcam: Use USB camera
        camera_index: USB camera index
        mean: Mean of Gaussian noise
        std: Standard deviation of Gaussian noise
    """
    # Initialize model
    denoise_model = DenoiseModel(model_path)
    
    input_w = denoise_model.input_width
    input_h = denoise_model.input_height
    
    # Division line position (for side-by-side comparison)
    div = input_w // 2
    
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
    
    # Create window
    cv2.namedWindow(DISPLAY_WINDOW_NAME)
    
    # Process based on input type
    if is_image:
        print(f"Processing image: {input_source}")
        frame = cv2.imread(input_source)
        if frame is None:
            print(f"Error: Could not read image {input_source}")
            return
        
        # Resize to model input size
        frame = cv2.resize(frame, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
        
        print("Controls:")
        print("  ESC - Exit")
        print("  1-8 - Change noise level (10-80)")
        print("  'a' - Move divider left")
        print("  'd' - Move divider right")
        
        while True:
            # Add noise
            noised_frame = get_noise_image(frame, mean, std)
            
            # Run inference
            start_time = time.time()
            output_frame = denoise_model.run(noised_frame)
            end_time = time.time()
            
            processing_time = (end_time - start_time) * 1000
            fps = 1000.0 / processing_time if processing_time > 0 else 0
            
            print("=" * 40)
            print(f"Processing time = {processing_time:.2f}ms, FPS = {fps:.2f}")
            
            # Create comparison view
            view = output_frame.copy()
            if div > 0:
                view[0:input_h, 0:div] = noised_frame[0:input_h, 0:div]
            
            # Draw divider line
            cv2.line(view, (div, 0), (div, input_h), (0, 0, 0), 2)
            
            # Add text info
            cv2.putText(view, f"Noise: {std:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(view, f"FPS: {fps:.1f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow(DISPLAY_WINDOW_NAME, view)
            
            key = cv2.waitKey(INPUT_CAPTURE_PERIOD_MS) & 0xFF
            
            if key == 27:  # ESC
                break
            elif ord('1') <= key <= ord('8'):  # Numbers 1-8
                std = (key - ord('0')) * 10.0
                print(f"Noise level changed to: {std}")
            elif key == ord('a'):  # Move divider left
                div -= 10
                if div < 0:
                    div = input_w // 2
            elif key == ord('d'):  # Move divider right
                div += 10
                if div > input_w:
                    div = input_w // 2
        
        cv2.destroyAllWindows()
    
    elif is_video or is_camera:
        # Setup video capture or camera
        if is_camera:
            camera, read_frame = setup_camera(
                use_rpicam=use_rpicam, 
                camera_index=camera_index,
                width=CAMERA_FRAME_WIDTH,
                height=CAMERA_FRAME_HEIGHT
            )
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
        
        print("Controls:")
        print("  ESC - Exit")
        print("  1-8 - Change noise level (10-80)")
        print("  'a' - Move divider left")
        print("  'd' - Move divider right")
        
        frame_count = 0
        
        while True:
            ret, frame = read_frame()
            
            if not ret:
                if is_video:
                    # Loop video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print("\nFailed to read frame.")
                    break
            
            # Resize frame
            resized_frame = cv2.resize(frame, (input_w, input_h), 
                                      interpolation=cv2.INTER_LINEAR)
            
            # Add noise
            noised_frame = get_noise_image(resized_frame, mean, std)
            
            # Run inference
            start_time = time.time()
            output_frame = denoise_model.run(noised_frame)
            end_time = time.time()
            
            processing_time = (end_time - start_time) * 1000
            fps = 1000.0 / processing_time if processing_time > 0 else 0
            
            frame_count += 1
            if frame_count % 30 == 0:
                print("=" * 40)
                print(f"Frame {frame_count} | Processing time = {processing_time:.2f}ms, FPS = {fps:.2f}")
            
            # Create comparison view
            view = output_frame.copy()
            if div > 0:
                view[0:input_h, 0:div] = noised_frame[0:input_h, 0:div]
            
            # Draw divider line
            cv2.line(view, (div, 0), (div, input_h), (0, 0, 0), 2)
            
            # Add text info
            cv2.putText(view, f"Noise: {std:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(view, f"FPS: {fps:.1f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow(DISPLAY_WINDOW_NAME, view)
            
            key = cv2.waitKey(INPUT_CAPTURE_PERIOD_MS) & 0xFF
            
            if key == 27:  # ESC
                break
            elif ord('1') <= key <= ord('8'):  # Numbers 1-8
                std = (key - ord('0')) * 10.0
                print(f"Noise level changed to: {std}")
            elif key == ord('a'):  # Move divider left
                div -= 10
                if div < 0:
                    div = input_w // 2
            elif key == ord('d'):  # Move divider right
                div += 10
                if div > input_w:
                    div = input_w // 2
        
        # Cleanup
        if is_camera:
            if use_rpicam and PICAMERA2_AVAILABLE:
                camera.stop()
            else:
                camera.release()
        else:
            cap.release()
        
        cv2.destroyAllWindows()
        
        print(f"\nTotal frames processed: {frame_count}")
    
    else:
        print("Error: No valid input source specified")
        print("Use --image, --video, --rpicam, or --usbcam")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DNCNN Denoise Demo with Camera Support')
    parser.add_argument('-m', '--model', required=True, type=str,
                        help='Path to dxnn model file')
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
    parser.add_argument('--mean', default=0.0, type=float,
                        help='Mean of Gaussian noise (default: 0.0)')
    parser.add_argument('--std', default=15.0, type=float,
                        help='Standard deviation of Gaussian noise (default: 15.0)')
    
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
    
    print("Starting DNCNN denoise app, please wait...")
    run_denoise(
        model_path=args.model,
        input_source=input_source,
        use_rpicam=args.rpicam,
        use_usbcam=args.usbcam,
        camera_index=args.camera_index,
        mean=args.mean,
        std=args.std
    )