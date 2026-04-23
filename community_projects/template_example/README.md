# Community Project Template

This repository serves as a standardized template for developing community projects within the DEEPX RPi5 Examples ecosystem. It provides a foundational structure for implementing Artificial Intelligence applications utilizing the DEEPX Neural Processing Unit (NPU).

## Project Information

- **Project Name**: Template Example
- **Category**: Template/Educational
- **Difficulty**: Beginner
- **Status**: Production Ready
- **Author**: Sixfab Community

## Description

This template illustrates the requisite structure and components for a DEEPX community project. It encompasses:

- DEEPX NPU integration
- Camera input acquisition
- Configuration management systems
- Error handling protocols
- Basic AI inference implementation
- Documentation standards

## Features

- **Multi-input Support**: Compatible with camera feeds, video files, and image sequences
- **DEEPX NPU Integration**: Optimized for DEEPX neural processing acceleration
- **Configurable Settings**: Customizable via external configuration files
- **Error Handling**: Comprehensive error management and user feedback systems
- **Performance Monitoring**: Real-time FPS and execution timing analytics
- **Comprehensive Logging**: Detailed logging for debugging and auditing

## Installation

### Prerequisites
- Raspberry Pi 5 with DEEPX NPU
- Python 3.9 or higher
- DEEPX runtime libraries

### Setup Procedure

1. **Clone and Navigate**
   ```bash
   cd deepx-rpi5-examples/community_projects/template_example
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Resource Acquisition** (Optional)
   ```bash
   ./download_resources.sh
   ```

## Usage

### Basic Execution
```bash
python template_example.py
```

### Advanced Execution configurations
```bash
# Select specific camera input
python template_example.py --input rpi

# Process video file
python template_example.py --input video.mp4

# Apply custom configuration
python template_example.py --config custom_config.yaml

# Enable debug mode
python template_example.py --debug
```

### Command Line Interface
```
--input SOURCE        Input source selection (auto/rpi/usb/file) [default: auto]
--config FILE         Configuration file path [default: config.yaml]
--model PATH          Custom model path [default: auto]
--output PATH         Directory for results output [default: output/]
--debug               Enable debug logging mode
--verbose             Enable verbose logging output
--no-display         Execute without display (headless capability)
--save-frames        Archive processed frames to output directory
--fps-limit N        Restrict execution to N frames per second
```

## Configuration

The application utilizes a YAML configuration file (`config.yaml`) for system customization:

```yaml
# Template Example Configuration
project:
  name: "Template Example"
  version: "1.0.0"
  
camera:
  resolution: [640, 480]
  fps: 30
  auto_exposure: true
  
processing:
  confidence_threshold: 0.5
  nms_threshold: 0.4
  max_detections: 100
  
display:
  show_fps: true
  show_confidence: true
  font_scale: 0.7
  
output:
  save_results: false
  result_format: "json"  # Options: json, csv, xml
  
deepx:
  model_path: "auto"
  device_id: 0
  batch_size: 1
  
logging:
  level: "INFO"  # Levels: DEBUG, INFO, WARNING, ERROR
  file: "template_example.log"
```

## Project Structure

```
template_example/
├── README.md                 # Project documentation
├── template_example.py       # Main application entry point
├── config.yaml              # System configuration
├── requirements.txt          # Dependency definitions
├── download_resources.sh     # Asset acquisition script
├── models/                   # Neural network models storage
├── resources/               # Auxiliary resources
├── output/                  # Generated artifacts directory
├── tests/                   # Test suite
│   ├── __init__.py
│   ├── test_template.py
│   └── test_config.py
└── docs/                    # Extended documentation
    └── architecture.md
```

## Testing

Execute the comprehensive test suite using the following commands:

```bash
# Execute all tests
python -m pytest tests/

# Execute specific test module
python -m pytest tests/test_template.py

# Execute with coverage report
python -m pytest tests/ --cov=template_example
```

## Performance Metrics

Anticipated performance benchmarks on Raspberry Pi 5 with DEEPX NPU:

| Configuration | FPS | Memory Usage | CPU Load |
|---------------|-----|--------------|----------|
| 640x480 @ 30fps | ~28-30 | ~150MB | ~15% |
| 1280x720 @ 30fps | ~20-25 | ~200MB | ~25% |
| 1920x1080 @ 30fps | ~12-15 | ~300MB | ~35% |

## Customization

### Feature Extension

1. **Extend the TemplateProcessor class**:
   ```python
   class CustomProcessor(TemplateProcessor):
       def custom_processing(self, frame):
           # Implementation of custom processing logic
           return processed_frame
   ```

2. **Append configuration options**:
   ```yaml
   custom:
     new_feature_enabled: true
     custom_parameter: 42
   ```

3. **Modify the main execution loop**:
   ```python
   def main():
       processor = CustomProcessor(config)
       # Custom initialization logic
   ```

### Model Integration

1. **Register model in configuration**:
   ```yaml
   deepx:
     model_path: "models/your_custom_model.dxnn"
   ```

2. **Implement model loading logic**:
   ```python
   def load_custom_model(self, model_path):
       # Model loading implementation
       pass
   ```

## Troubleshooting

### Common Issues

#### 1. NPU Detection Failure
```bash
# Verify NPU status
ls /dev/dxrt0*
lsmod | grep dx

# Resolution: Reload DEEPX kernel module
sudo modprobe dx_dma
sudo modprobe dxrt_driver
```

#### 2. Camera Initialization Failure
```bash
# Enumerate available devices
v4l2-ctl --list-devices

# Diagnostic test
v4l2-ctl --device=/dev/video0 --all
```

#### 3. Performance Degradation
```bash
# Monitor system resources
htop
free -h

# Activate performance governor
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

## Contribution Guidelines

### Development Workflow
1. Fork the repository
2. Create a specific feature branch
3. Implement changes
4. Incorporate tests for new functionality
5. Update documentation
6. Submit a Pull Request

### Code Standards
- Adhere to PEP 8 style guidelines
- Utilize type hinting
- Include docstrings for all functions
- Maintain modular and focused function design

### Testing Requirements
- Implement unit tests for new features
- Verify all tests pass
- Validate on Raspberry Pi 5 hardware
- Assess performance impact

## Support

- **Issue Tracking**: Report defects via [GitHub Issues](https://github.com/sixfab/deepx-rpi5-examples/issues)
- **Discussions**: Engage in [GitHub Discussions](https://github.com/sixfab/deepx-rpi5-examples/discussions)
- **Documentation**: Consult the [official documentation](../../docs/)