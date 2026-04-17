#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[Start] Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libopencv-dev \
    nlohmann-json3-dev \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad

# libcamera is only on Raspberry Pi OS; skip silently if missing.
if apt-cache show libcamera-dev >/dev/null 2>&1; then
    sudo apt-get install -y libcamera-dev gstreamer1.0-libcamera || true
fi

echo "[Sixfab] Building..."
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j"$(nproc)"

echo
echo "[Sixfab] Build complete."
echo "         Run a demo from the build directory, e.g.:"
echo "         ./start.sh"
