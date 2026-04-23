#!/bin/bash
# Template Example Resource Download Script

echo "Downloading resources for Template Example..."

# Create directories
mkdir -p models
mkdir -p resources
mkdir -p output

# Download placeholder model (replace with actual model URLs when available)
echo "Setting up model directory..."

# Placeholder model file
cat > models/README.md << 'EOF'
# Template Example Models

Place your DEEPX model files (.dxnn) in this directory.

## Model Requirements
- Format: .dxnn (DEEPX NPU optimized format)  
- Input: RGB image, typically 640x640
- Output: Detection boxes, confidence scores, class labels

## Download Models
Replace this file with actual model download commands when models are available.

Example:
```bash
wget -O models/yolov5s.dxnn "https://example.com/models/yolov5s.dxnn"
```

## Model Conversion
If you have ONNX models, convert them using DEEPX compiler:
```bash
dx_compiler --input model.onnx --output models/model.dxnn
```
EOF

# Create sample resource files
echo "Setting up sample resources..."

cat > resources/classes.txt << 'EOF'
person
bicycle
car
motorcycle
airplane
bus
train
truck
boat
traffic light
fire hydrant
stop sign
parking meter
bench
bird
cat
dog
horse
sheep
cow
elephant
bear
zebra
giraffe
backpack
umbrella
handbag
tie
suitcase
frisbee
skis
snowboard
sports ball
kite
baseball bat
baseball glove
skateboard
surfboard
tennis racket
bottle
wine glass
cup
fork
knife
spoon
bowl
banana
apple
sandwich
orange
broccoli
carrot
hot dog
pizza
donut
cake
chair
couch
potted plant
bed
dining table
toilet
tv
laptop
mouse
remote
keyboard
cell phone
microwave
oven
toaster
sink
refrigerator
book
clock
vase
scissors
teddy bear
hair drier
toothbrush
EOF

echo "Template resources setup completed!"
echo ""
echo "Next steps:"
echo "1. Add your DEEPX model files to models/ directory"
echo "2. Update config.yaml with your model path"
echo "3. Run: python template_example.py"
echo ""
echo "For more information, see README.md"