#!/bin/bash

# Sixfab | DEEPX Raspberry Pi 5 Automatic Installation Script
# This script installs DEEPX NPU drivers and runtime on Raspberry Pi 5
# Author: Sixfab | DEEPX Raspberry Pi 5 Examples
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

# Check DEEPX NPU runtime first
print_info "Checking DEEPX NPU runtime..."

if ! command_exists dxrt-cli; then
    print_error "DEEPX NPU runtime is not installed."
    echo ""
    echo "  Please install the sixfab-dx package first to use the demos and DEEPX NPU runtime."
    echo ""
    exit 1
fi

if ! dxrt-cli -s > /dev/null 2>&1; then
    print_error "DEEPX NPU runtime is installed but not working properly."
    echo ""
    echo "  Please make sure the sixfab-dx package is correctly installed and the NPU is recognized."
    echo ""
    exit 1
fi

print_success "DEEPX NPU runtime is working"

# Check other prerequisites
print_info "Checking prerequisites..."

if ! command_exists python3; then
    print_error "Python3 is not installed. Please install Python3 first."
    exit 1
fi

if ! command_exists curl; then
    print_error "curl is not installed. Please install curl first."
    exit 1
fi

if ! command_exists tar; then
    print_error "tar is not installed. Please install tar first."
    exit 1
fi

if ! command_exists sed; then
    print_error "sed is not installed. Please install sed first."
    exit 1
fi

print_success "Prerequisites check completed"

# Set working directory
VENV_PATH="/opt/sixfab-dx/venv"
CURRENT_DIR="$(pwd)"

print_info "Starting DEEPX examples setup..."

# Step 0: Download and extract resources
print_info "Step 0: Checking resources (images, models, videos)..."
RESOURCES_DIR="$CURRENT_DIR/resources"
mkdir -p "$RESOURCES_DIR"

for asset in images models videos; do
    ASSET_DIR="$RESOURCES_DIR/$asset"
    if [ -d "$ASSET_DIR" ] && [ "$(ls -A "$ASSET_DIR" 2>/dev/null)" ]; then
        print_warning "Skipping $asset — already exists in $ASSET_DIR"
        continue
    fi

    url="https://github.com/sixfab/deepx-rpi5-examples/releases/download/v0.3/${asset}.tar.gz"
    print_info "Downloading ${asset}.tar.gz..."
    curl -L --retry 3 --retry-delay 5 --max-time 300 \
        --progress-bar \
        -o "$RESOURCES_DIR/${asset}.tar.gz" \
        "$url"

    if [ ! -s "$RESOURCES_DIR/${asset}.tar.gz" ]; then
        print_error "Download failed or file is empty: ${asset}.tar.gz"
        exit 1
    fi

    print_info "Extracting ${asset}.tar.gz..."
    tar -xzf "$RESOURCES_DIR/${asset}.tar.gz" -C "$RESOURCES_DIR"
    rm "$RESOURCES_DIR/${asset}.tar.gz"
    print_success "${asset} downloaded and extracted"
done

print_success "Resources ready at $RESOURCES_DIR"

# Step 1: Activate existing virtual environment
print_info "Step 1: Activating virtual environment at $VENV_PATH..."

if [ ! -f "$VENV_PATH/bin/activate" ]; then
    print_error "Virtual environment not found at $VENV_PATH"
    echo ""
    echo "  Please make sure the sixfab-dx package is correctly installed."
    echo ""
    exit 1
fi

source "$VENV_PATH/bin/activate"
print_success "Virtual environment activated"

# Step 2: Install Python examples requirements
print_info "Step 2: Installing Python requirements..."
cd "$CURRENT_DIR"
pip install -r python_examples/requirements.txt
print_success "Python requirements installed"

# Step 3: Build C++ examples
print_info "Step 3: Building C++ examples..."
cd "$CURRENT_DIR/cpp_examples"
bash setup.sh
cd "$CURRENT_DIR"
print_success "C++ examples built successfully"

# Final instructions
echo ""
print_info "=== Installation Summary ==="
print_success "DEEPX examples have been successfully set up!"
echo ""
print_info "To activate the environment in future sessions, run:"
echo "  source $VENV_PATH/bin/activate"
echo ""
print_info "To start the Python examples:"
echo "  cd $CURRENT_DIR && bash python_examples/start.sh"
echo ""
print_info "To start the C++ examples:"
echo "  cd $CURRENT_DIR && bash cpp_examples/start.sh"
echo ""
print_info "For examples and documentation, check:"
echo "  - python_examples/ directory"
echo "  - cpp_examples/ directory"
echo ""
print_success "Happy coding!"
