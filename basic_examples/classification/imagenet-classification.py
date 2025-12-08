import os
import cv2
import numpy as np
import json
import argparse
import time
import sys
from threading import Thread, Lock
from dx_engine import InferenceEngine, Configuration
from packaging import version

# Try to import picamera2 for Raspberry Pi camera support
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    print("Warning: picamera2 not available. Raspberry Pi camera support disabled.")

# Constants
DEFAULT_MODEL_PATH = "/dxrt/m1/efficientnet-b0_argmax"
DEFAULT_IMAGE_PATH = "/dxrt/m1/imagenet/imagenet_val/"
DEFAULT_LABEL_PATH = "/dxrt/m1/imagenet/imagenet_val.json"
DEFAULT_GRID_PATH = "/dxrt/m1/imagenet/grid_8x5/"

NUM_IMAGES = 50000
NUM_BUFFS = 500
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
GRID_WIDTH = 8
GRID_HEIGHT = 5
GRID_UNIT = GRID_WIDTH * GRID_HEIGHT

# For visualization
MODEL_NAME = "EfficientNetB0"
CHIP_NAME = "DX-M1"
TOPS = 23.0

class ImageNetClassifier:
    def __init__(self, model_path, image_path, label_path, grid_path):
        self.model_path = model_path
        self.image_path = image_path
        self.label_path = label_path
        self.grid_path = grid_path
        
        # Initialize engine
        self.ie = InferenceEngine(model_path)
        if version.parse(self.ie.get_model_version()) < version.parse('7'):
            print("dxnn files format version 7 or higher is required. Please update/re-export the model.")
            sys.exit(1)
        
        # Statistics
        self.ground_truth = np.zeros(NUM_IMAGES, dtype=np.int32)
        self.classification = np.zeros(NUM_IMAGES, dtype=np.int32)
        self.results = np.zeros(NUM_IMAGES, dtype=bool)
        self.result_lock = Lock()
        
        self.grid_idx = 0
        self.inf_cnt = 0
        self.correct = 0
        self.accuracy = 0.0
        self.fps = 0.0
        self.exit_flag = False
        
        # Load ground truth
        self.load_ground_truth()
        
        # Preprocess all images
        print("Preprocessing images...")
        self.inputs = self.preprocess_all()
        
        # Load grid images
        print("Loading grid images...")
        self.grids = self.load_grids()

    def get_imagenet_name(self, index):
        return f"ILSVRC2012_val_{index+1:08d}"

    def preprocess(self, image_index):
        """Preprocess a single image"""
        name = self.get_imagenet_name(image_index)
        image_file = os.path.join(self.image_path, name + ".PNG")
        
        if not os.path.exists(image_file):
            image_file = os.path.join(self.image_path, name + ".JPEG")
        
        image = cv2.imread(image_file)
        if image is None:
            print(f"Error: Cannot load image {image_file}")
            return None
        
        # Resize if needed
        if image.shape[1] != IMAGE_WIDTH or image.shape[0] != IMAGE_HEIGHT:
            image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def preprocess_all(self):
        """Preprocess all images"""
        inputs = []
        for i in range(NUM_IMAGES):
            if i % 1000 == 0:
                print(f"Preprocessing: {i}/{NUM_IMAGES}")
            
            img = self.preprocess(i)
            if img is not None:
                inputs.append(img)
            else:
                # Use dummy image if loading fails
                inputs.append(np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8))
        
        return inputs

    def load_ground_truth(self):
        """Load ground truth labels from JSON"""
        print("Loading ground truth labels...")
        with open(self.label_path, 'r') as f:
            label_data = json.load(f)
        
        for i in range(NUM_IMAGES):
            dataset_name = self.get_imagenet_name(i)
            if dataset_name in label_data:
                self.ground_truth[i] = label_data[dataset_name]

    def load_grids(self):
        """Load grid visualization images"""
        num_grids = NUM_IMAGES // GRID_UNIT
        grids = []
        for i in range(num_grids):
            grid_file = os.path.join(self.grid_path, f"{i:05d}.JPEG")
            grid = cv2.imread(grid_file)
            if grid is None:
                print(f"Warning: Cannot load grid {grid_file}")
                grid = np.zeros((IMAGE_HEIGHT * GRID_HEIGHT, IMAGE_WIDTH * GRID_WIDTH, 3), dtype=np.uint8)
            grids.append(grid)
        return grids

    def post_process_callback(self, outputs, image_id):
        """Callback for processing inference results"""
        with self.result_lock:
            self.inf_cnt = image_id
            
            # Get classification result (argmax output)
            self.classification[image_id] = int(outputs[0].flatten()[0])
            
            # Check if correct
            if self.classification[image_id] == self.ground_truth[image_id]:
                self.correct += 1
                self.results[image_id] = True
                self.accuracy = self.correct / (image_id + 1)
            
            # Update grid index
            if image_id % GRID_UNIT == GRID_UNIT - 1:
                self.grid_idx = image_id // GRID_UNIT

    def make_board(self, count, accuracy, fps, board_height):
        """Create information board for visualization"""
        acc = accuracy * 100
        
        board = np.full((board_height, 400, 3), (179, 102, 0), dtype=np.uint8)
        
        font = cv2.FONT_HERSHEY_COMPLEX
        color = (255, 255, 255)
        
        # Labels
        cv2.putText(board, "ImageNet 2012", (32, 250), font, 1.0, color, 2, cv2.LINE_AA)
        cv2.putText(board, "Accuracy (%)", (32, 450), font, 1.0, color, 2, cv2.LINE_AA)
        cv2.putText(board, "Frame Rate (fps)", (32, 650), font, 1.0, color, 2, cv2.LINE_AA)
        
        # Values
        cv2.putText(board, f" {count}", (32, 340), font, 2.5, color, 6, cv2.LINE_AA)
        cv2.putText(board, f" {acc:.2f}", (32, 540), font, 2.5, color, 6, cv2.LINE_AA)
        cv2.putText(board, f" {fps:.0f}", (32, 740), font, 2.5, color, 6, cv2.LINE_AA)
        
        # Model info
        cv2.putText(board, MODEL_NAME, (32, 850), font, 1.0, color, 2, cv2.LINE_AA)
        cv2.putText(board, f"{IMAGE_WIDTH} x {IMAGE_HEIGHT}", (32, 900), font, 1.0, color, 2, cv2.LINE_AA)
        cv2.putText(board, CHIP_NAME, (32, 950), font, 1.0, color, 2, cv2.LINE_AA)
        cv2.putText(board, f"{fps/TOPS:.2f} FPS/TOPS", (32, 1000), font, 1.0, color, 2, cv2.LINE_AA)
        
        return board

    def inference_thread(self):
        """Thread for running inference"""
        print("Starting inference thread...")
        cnt = 0
        self.correct = 0
        
        start_time = time.time()
        batch_start = time.time()
        
        while not self.exit_flag and cnt < NUM_IMAGES:
            # Run inference synchronously
            outputs = self.ie.run([self.inputs[cnt]])
            
            # Process result
            self.post_process_callback(outputs, cnt)
            
            cnt += 1
            
            # Update FPS every batch
            if cnt % NUM_BUFFS == 0:
                batch_end = time.time()
                batch_time = batch_end - batch_start
                self.fps = NUM_BUFFS / batch_time
                batch_start = time.time()
                print(f"Processed: {cnt}/{NUM_IMAGES}, Accuracy: {self.accuracy*100:.2f}%, FPS: {self.fps:.1f}")
        
        total_time = time.time() - start_time
        print(f"\nInference completed!")
        print(f"Total images: {cnt}")
        print(f"Correct: {self.correct}")
        print(f"Final Accuracy: {self.accuracy*100:.2f}%")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average FPS: {cnt/total_time:.1f}")

    def visualization_thread(self, window_name):
        """Thread for visualization"""
        print("Starting visualization thread...")
        
        while not self.exit_flag:
            grid_index = self.grid_idx
            count = self.inf_cnt
            
            if grid_index < len(self.grids):
                # Create grid visualization
                grid = self.grids[grid_index].copy()
                
                # Draw rectangles for each image in grid
                for index in range(GRID_UNIT):
                    gx = (index % GRID_WIDTH) * IMAGE_WIDTH
                    gy = (index // GRID_WIDTH) * IMAGE_HEIGHT
                    
                    # Determine color based on result
                    result_idx = grid_index * GRID_UNIT + index
                    if result_idx < NUM_IMAGES and self.results[result_idx]:
                        color = (64, 255, 0)  # Green for correct
                    else:
                        color = (0, 64, 255)  # Red for incorrect/pending
                    
                    cv2.rectangle(grid, 
                                (gx + 4, gy + 4), 
                                (gx + IMAGE_WIDTH - 4, gy + IMAGE_HEIGHT - 4), 
                                color, 4)
                
                # Create board
                board = self.make_board(count + 1, self.accuracy, self.fps, grid.shape[0])
                
                # Combine board and grid
                view = np.hstack([board, grid])
                
                # Display
                cv2.imshow(window_name, view)
            
            key = cv2.waitKey(1)
            if key == 27:  # ESC key
                self.exit_flag = True
                break
        
        cv2.destroyAllWindows()

    def run(self):
        """Main execution method"""
        window_name = "ImageNet Classification"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.resizeWindow(window_name, 1920, 1080)
        
        # Start threads
        inf_thread = Thread(target=self.inference_thread)
        vis_thread = Thread(target=self.visualization_thread, args=(window_name,))
        
        inf_thread.start()
        vis_thread.start()
        
        # Wait for threads
        inf_thread.join()
        vis_thread.join()
        
        print("Application finished!")


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


def run_camera_classification(model_path, use_rpicam=False, camera_index=0):
    """Run real-time classification on camera feed"""
    print("Initializing camera classification...")
    
    # Setup camera
    camera, read_frame = setup_camera(
        use_rpicam=use_rpicam, 
        camera_index=camera_index,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        fps=30
    )
    
    # Initialize engine
    ie = InferenceEngine(model_path)
    
    window_name = "ImageNet Classification - Camera"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    print("Press 'q' to quit...")
    
    frame_times = []
    
    try:
        while True:
            start_time = time.time()
            
            ret, frame = read_frame()
            if not ret:
                print("Failed to read frame")
                break
            
            # Preprocess
            frame_resized = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
            input_image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # Run inference
            outputs = ie.run([input_image])
            class_id = int(outputs[0].flatten()[0])
            
            # Calculate FPS
            end_time = time.time()
            frame_time = end_time - start_time
            frame_times.append(frame_time)
            if len(frame_times) > 30:
                frame_times.pop(0)
            avg_fps = len(frame_times) / sum(frame_times)
            
            # Display result
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Class: {class_id}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, f"FPS: {avg_fps:.1f}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow(window_name, display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Cleanup
        if use_rpicam and PICAMERA2_AVAILABLE:
            camera.stop()
        else:
            camera.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='ImageNet Classification with DX Engine')
    parser.add_argument('--model', default=DEFAULT_MODEL_PATH, type=str,
                       help='Path to the model file')
    parser.add_argument('--images', default=DEFAULT_IMAGE_PATH, type=str,
                       help='Path to ImageNet validation images')
    parser.add_argument('--labels', default=DEFAULT_LABEL_PATH, type=str,
                       help='Path to ground truth labels JSON')
    parser.add_argument('--grids', default=DEFAULT_GRID_PATH, type=str,
                       help='Path to grid visualization images')
    parser.add_argument('--rpicam', action='store_true',
                       help='Use Raspberry Pi camera for real-time classification')
    parser.add_argument('--usbcam', action='store_true',
                       help='Use USB camera for real-time classification')
    parser.add_argument('--camera_index', default=0, type=int,
                       help='USB camera index (default: 0)')
    
    args = parser.parse_args()
    
    # Check DX-RT version
    if version.parse(Configuration().get_version()) < version.parse("3.0.0"):
        print("DX-RT version 3.0.0 or higher is required. Please update DX-RT to the latest version.")
        sys.exit(1)
    
    # Validate arguments
    if args.rpicam and args.usbcam:
        print("Error: Cannot use both --rpicam and --usbcam at the same time")
        sys.exit(1)
    
    # Run camera mode or batch mode
    if args.rpicam or args.usbcam:
        run_camera_classification(args.model, use_rpicam=args.rpicam, camera_index=args.camera_index)
    else:
        # Check if paths exist
        if not os.path.exists(args.model):
            print(f"Error: Model file not found: {args.model}")
            sys.exit(1)
        if not os.path.exists(args.images):
            print(f"Error: Images directory not found: {args.images}")
            sys.exit(1)
        if not os.path.exists(args.labels):
            print(f"Error: Labels file not found: {args.labels}")
            sys.exit(1)
        
        # Run batch classification
        classifier = ImageNetClassifier(args.model, args.images, args.labels, args.grids)
        classifier.run()


if __name__ == "__main__":
    main()