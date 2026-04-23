# DEEPX — Raspberry Pi 5 Examples

Compact, practical AI inference examples for the Raspberry Pi 5 using the DEEPX NPU. Examples range from object detection and pose estimation to instance segmentation and image denoising — designed to get you running on edge AI quickly.

**Quick links**
- [Sixfab Website](https://sixfab.com) | [Sixfab Docs](https://docs.sixfab.com)
- [DEEPX Website](https://deepx.ai)
- Runtime & tools: [sixfab-dx](https://github.com/sixfab/sixfab_dx) [dx_rt](https://github.com/DEEPX-AI/dx_rt)
- Driver: [dx_rt_npu_linux_driver](https://github.com/DEEPX-AI/dx_rt_npu_linux_driver)
- Full toolchain: [dx-all-suite](https://github.com/DEEPX-AI/dx-all-suite)

---

## Repository Layout

```
deepx-rpi5-examples/
├── auto-install.sh           # One-shot setup: resources, Python deps, C++ build
├── python_examples/          # Full Python demo collection (28 demos, TUI launcher)
├── cpp_examples/             # Full C++ demo collection (26 demos, TUI launcher)
├── community_projects/       # Community contributions and templates
└── docs/                     # Guides for installation and usage
```

---

## Getting Started

### Prerequisites

Install the DEEPX runtime first via the Sixfab APT repository:

```bash
# Add the Sixfab GPG key
wget -qO - https://sixfab.github.io/sixfab_dx/public.gpg | sudo gpg --dearmor -o /usr/share/keyrings/sixfab-dx.gpg

# Add the APT repository
echo "deb [signed-by=/usr/share/keyrings/sixfab-dx.gpg] https://sixfab.github.io/sixfab_dx trixie main" | sudo tee /etc/apt/sources.list.d/sixfab-dx.list

# Install sixfab-dx (runtime + Python wheels + pre-built venv)
sudo apt update && sudo apt install sixfab-dx

# Verify the NPU is recognized
dxrt-cli -s
```

See [docs/install-raspberry-pi5.md](docs/install-raspberry-pi5.md) for full hardware and software setup instructions.

### Quick Setup

Once `sixfab-dx` is installed, run the setup script to download models and build the C++ examples:

```bash
git clone https://github.com/sixfab/deepx-rpi5-examples.git
cd deepx-rpi5-examples
./auto-install.sh
```

`auto-install.sh` will:
1. Validate the DEEPX runtime is working
2. Download models, videos, and sample images from the GitHub release
3. Install Python dependencies into the Sixfab venv
4. Build all C++ demos

After setup, activate the environment for every new session:

```bash
source /opt/sixfab-dx/venv/bin/activate
```

---

## Example Collections

### Python Examples (`python_examples/`)

A full collection of 28 demos with a Rich-based TUI launcher. Covers object detection (13 YOLO variants, SCRFD, YOLOX), segmentation, classification, pose estimation, PPU-accelerated variants, and advanced analytics (zone intrusion, people tracking, traffic counting, queue analysis).

```bash
# Launch the interactive menu
bash python_examples/start.sh
```

See [docs/python-examples.md](docs/python-examples.md) for the full demo list and configuration guide.

### C++ Examples (`cpp_examples/`)

26 demos written in idiomatic C++17 using only OpenCV and the DeepX SDK — no Qt, no YAML. Same categories as the Python collection. Each demo is a single readable file.

```bash
# Launch the interactive menu
bash cpp_examples/start.sh

# Or run a binary directly
./cpp_examples/build/object_detection/yolov8_demo --source libcamera
```

See [docs/cpp-examples.md](docs/cpp-examples.md) for the demo list, build instructions, and flags.

---

## Input Sources

All demos accept the same set of input sources:

| Source | config.yml value |
|--------|-----------------|
| Raspberry Pi camera | `rpicam` |
| USB webcam | `webcam` |
| Video file | `video` |
| Single image | `image` |
| RTSP stream | `rtsp` |

---

## Community and Contribution

Community improvements are welcome. If you have a demo, model optimization tip, or pipeline to share, add it under `community_projects/` and update `community_projects/community_projects.md` with usage notes. See the template at `community_projects/template_example/` for the expected structure.

---

## License

This project is distributed under the **MIT License**.
