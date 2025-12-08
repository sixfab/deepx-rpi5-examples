# DEEPX Raspberry Pi 5 Installation Guide

This guide will help you set up DEEPX NPU hardware and software on Raspberry Pi 5.

## Table of Contents
- [Hardware Requirements](#hardware-requirements)
- [Software Prerequisites](#software-prerequisites)
- [Installation](#installation)
- [Verification and Testing](#verification-and-testing)
- [Additional Resources](#additional-resources)

## Hardware Requirements

### Supported Hardware
- **Raspberry Pi 5** (8GB RAM recommended)
- **DEEPX NPU Module** (compatible with RPi5)
- **MicroSD Card** (32GB or larger, Class 10 or better)
- **Power Supply** (Official RPi5 power adapter recommended)
- **Camera** (RPi Camera Module or USB webcam)

### Optional Hardware
- Heat sinks or cooling fan for better performance
- External storage (USB drive/SSD)

## Software Prerequisites

### Operating System
- **Raspberry Pi OS** (64-bit, Bookworm recommended)

### System Requirements
```bash
# Check your system
uname -a
lscpu
free -h
```

### Update System
```bash
sudo apt update
sudo apt upgrade -y
sudo reboot
```

### Install Basic Dependencies
```bash
sudo apt install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv
```

## Installation

### Method 1: Alternative Automatic Installation

Clone this repository and run the installation script:

```bash
git clone https://github.com/sixfab/deepx-rpi5-examples.git
cd deepx-rpi5-examples
./auto-install.sh
```

This will automatically:
1. Download and install DEEPX NPU driver
2. Set up DEEPX Runtime
3. Create Python virtual environment
4. Install all dependencies
5. Download example models

### Method 2: Install with DeepX scripts

Follow these tested steps for successful installation:

```bash
# Create virtual environment with system site packages
python3 -m venv --system-site-packages deepx
source deepx/bin/activate

# Download and install NPU driver
cd /home/$USER
git clone https://github.com/DEEPX-AI/dx_rt_npu_linux_driver
cd /home/$USER/dx_rt_npu_linux_driver
./install.sh

# Verify driver installation
lsmod | grep dx
cd /home/$USER

# Install DEEPX runtime
git clone https://github.com/DEEPX-AI/dx_rt
cd /home/$USER/dx_rt

./install.sh --all
./build.sh

# Install Python package
pip install . /home/$USER/dx_rt/python_package
```

### Method 3: Official DEEPX Suite

For complete development environment, follow the official guide:

```bash
git clone --recurse-submodules https://github.com/DEEPX-AI/dx-all-suite.
cd dx-all-suite

python3 -m venv --system-site-packages deepx
source deepx/bin/activate
./dx-runtime/install.sh --all
```

## Verification and Testing

### Check Hardware Detection
```bash
# Check kernel modules
lsmod | grep dx

# Check system logs
dmesg | grep -i dx
```

### Test DEEPX Installation
```bash
# Activate environment (if using the working method)
source deepx/bin/activate

dxrt-cli -s
```

### Getting Help

1. **Check System Logs**
   ```bash
   sudo journalctl -u dxrtd
   dmesg | grep -i dx
   lsmod | grep dx # Check driver
   ```

2. **Community Support**
   - Sixfab Community Support: https://github.com/sixfab/deepx-rpi5-examples
   - DEEPX Official Documentation: https://github.com/DEEPX-AI/dx-all-suite

## Next Steps

After successful installation:

1. **Explore Examples**: Try different examples in `basic_examples/`
2. **Read Documentation**: Check `docs/basic-examples.md` for detailed usage
3. **Join Community**: Contribute to community projects in `community_projects/`
4. **Optimize Performance**: Experiment with model optimization and hardware settings

## Additional Resources

- [DEEPX Official Website](https://www.deepx.ai/)
- [dx_rt Repository](https://github.com/DEEPX-AI/dx_rt)
- [dx_rt_npu_linux_driver Repository](https://github.com/DEEPX-AI/dx_rt_npu_linux_driver)
- [dx-all-suite Repository](https://github.com/DEEPX-AI/dx-all-suite)

---

*This installation guide is maintained by Sixfab. If you encounter issues or have suggestions, please open an issue on GitHub.*