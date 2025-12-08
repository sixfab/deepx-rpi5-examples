#!/bin/bash

# DEEPX Raspberry Pi 5 Automatic Installation Script
# This script installs DEEPX NPU drivers and runtime on Raspberry Pi 5
# Author: Sixfab Community | DEEPX Raspberry Pi 5 Examples
# Version: 1.0

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
print_info "Checking prerequisites..."

if ! command_exists python3; then
    print_error "Python3 is not installed. Please install Python3 first."
    exit 1
fi

if ! command_exists git; then
    print_error "Git is not installed. Please install Git first."
    exit 1
fi

if ! command_exists wget; then
    print_error "wget is not installed. Please install wget first."
    exit 1
fi

print_success "Prerequisites check completed"

# Set working directory
WORK_DIR="/home/$USER"
VENV_NAME="deepx"
VENV_PATH="$WORK_DIR/$VENV_NAME"
CURRENT_DIR="$(pwd)"

print_info "Starting DEEPX installation process..."
print_info "Working directory: $WORK_DIR"
print_info "Virtual environment: $VENV_PATH"

cd $CURRENT_DIR/basic_examples/src/models/
wget https://github.com/sixfab/deepx-rpi5-examples/releases/download/v0.1/sample_models.zip
unzip sample_models.zip
rm sample_models.zip

cd $CURRENT_DIR

# Step 1: Create Python virtual environment
print_info "Step 1: Creating Python virtual environment with system site packages..."
cd "$WORK_DIR"
python3 -m venv --system-site-packages "$VENV_NAME"

# Activate virtual environment
print_info "Activating virtual environment..."
source "$VENV_PATH/bin/activate"
print_success "Virtual environment activated"

# Step 2: Install DEEPX NPU Linux Driver
print_info "Step 2: Installing DEEPX NPU Linux Driver..."
cd "$WORK_DIR"

if [ -d "dx_rt_npu_linux_driver" ]; then
    print_warning "Driver directory already exists. Removing old installation..."
    rm -rf dx_rt_npu_linux_driver
fi

git clone https://github.com/DEEPX-AI/dx_rt_npu_linux_driver
cd "$WORK_DIR/dx_rt_npu_linux_driver"

print_info "Running driver installation script..."
echo "n" | ./install.sh

print_info "Verifying driver installation..."
if lsmod | grep -q dx; then
    print_success "DEEPX driver loaded successfully"
    lsmod | grep dx
else
    print_warning "DEEPX driver not found in loaded modules"
fi

# Step 3: Install DEEPX Runtime
print_info "Step 3: Installing DEEPX Runtime..."
cd "$WORK_DIR"

if [ -d "dx_rt" ]; then
    print_warning "Runtime directory already exists. Removing old installation..."
    rm -rf dx_rt
fi

git clone https://github.com/DEEPX-AI/dx_rt
cd "$WORK_DIR/dx_rt"

print_info "Running runtime installation script..."
./install.sh --all

print_info "Building DEEPX runtime..."
./build.sh

# Step 4: Install Python package
print_info "Step 4: Installing DEEPX Python package..."
cd "$WORK_DIR/dx_rt/python_package"
pip install .

print_success "DEEPX installation completed successfully!"

# Step 5: Verification
print_info "Step 5: Verifying installation..."
python3 -c "
try:
    import dx_engine
    print('✅ DEEPX Runtime imported successfully')
except ImportError as e:
    print('❌ DEEPX Runtime import failed:', e)
    exit(1)
"

print_success "Installation verification completed"

# Step 6: Install DEEPX Official Examples
print_info "Step 6: DEEPX Official Examples (dx_app)"
echo ""
read -p "Do you want to install dx_app (DEEPX Examples)? (yes/no): " install_choice

if [[ "$install_choice" =~ ^[Yy][Ee][Ss]$ ]] || [[ "$install_choice" =~ ^[Yy]$ ]]; then
    print_info "Installing dx_app to $WORK_DIR directory..."
    cd "$WORK_DIR"
    
    if [ -d "dx_app" ]; then
        print_warning "dx_app directory already exists. Removing old installation..."
        rm -rf dx_app
    fi
    
    git clone https://github.com/DEEPX-AI/dx_app
    cd dx_app
    
    print_info "Installing Python requirements..."
    pip install -r ./templates/python/requirements.txt
    
    print_info "Running dx_app installation script..."
    ./install.sh --all
    
    print_info "Building dx_app..."
    ./build.sh
    
    print_info "Setting up dx_app..."
    ./setup.sh
    
    print_success "dx_app installation completed in $WORK_DIR/dx_app!"
else
    print_info "Skipping dx_app installation."
fi

# Final instructions
echo ""
print_info "=== Installation Summary ==="
print_success "DEEPX has been successfully installed!"
echo ""
print_info "To activate the environment in future sessions, run:"
echo "  source $VENV_PATH/bin/activate"
echo ""
print_info "To test your installation, try running:"
echo "  python3 -c 'import dx_engine; print(\"DEEPX Ready!\")'"
echo ""
print_info "For examples and documentation, check:"
echo "  - basic_examples/ directory"
echo "  - basic_examples/deepx-official-examples/dx_app/ directory"
echo "  - docs/basic-examples.md"
echo ""
print_success "Happy coding!"
