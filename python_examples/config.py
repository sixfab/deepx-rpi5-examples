"""Central configuration for all DeepX demos.

Every demo imports this module and uses the values below as defaults. Any
value can still be overridden from the command line via argparse, so editing
this file is purely for convenience when running the same demo repeatedly.
"""

# --- Input Source --------------------------------------------------------
# Where frames come from. One of: "webcam", "rpicam", "video", "image".
INPUT_SOURCE: str = "webcam"

# Index of the USB / built-in camera (used only when INPUT_SOURCE == "webcam").
WEBCAM_INDEX: int = 0

# Path to a video file (used only when INPUT_SOURCE == "video").
VIDEO_PATH: str = "input.mp4"

# Path to a single image (used only when INPUT_SOURCE == "image").
# The InputSource will keep returning this same image so that the main loop
# can still run normally — useful for benchmarking or quick visual checks.
IMAGE_PATH: str = "input.jpg"

# --- Model ---------------------------------------------------------------
# Path to the compiled DeepX model. The runtime expects a .dxnn file
# produced by DX-COM. The variable is named MODEL_PATH for clarity, but
# the file extension itself does not have to be ".npu".
MODEL_PATH: str = "model.dxnn"

# Optional class-label file. One label per line. If a demo uses a built-in
# label set (e.g. COCO80, ImageNet1000), it will fall back to that when
# the file does not exist on disk.
LABEL_PATH: str = "labels.txt"

# --- Inference -----------------------------------------------------------
# Drop detections with confidence below this threshold.
CONFIDENCE_THRESHOLD: float = 0.3

# Intersection-over-Union threshold used by Non-Maximum Suppression to
# merge overlapping boxes that point at the same object.
IOU_THRESHOLD: float = 0.45

# --- Display -------------------------------------------------------------
# Overlay an FPS counter on the top-left of every frame.
SHOW_FPS: bool = True

# Title shown in the OpenCV display window.
WINDOW_NAME: str = "DeepX Demo"

# Print the class name next to every detection box.
DRAW_LABELS: bool = True

# Print the confidence score next to every detection box.
DRAW_CONFIDENCE: bool = True

# --- PPU -----------------------------------------------------------------
# PPU (Post Processing Unit) is a hardware accelerator inside the NPU
# that performs the final detection decoding step. When the model was
# compiled with PPU support, the .dxnn file emits a compact byte-encoded
# tensor instead of raw network outputs — much faster, but requires a
# different postprocess routine. The PPU demos in ppu/ flip this on.
PPU_ENABLED: bool = False
