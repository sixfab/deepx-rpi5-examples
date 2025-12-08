# DEEPX — Raspberry Pi 5 Examples

This repository contains compact, practical examples that demonstrate how to run AI workloads on the Raspberry Pi 5 using the DEEPX NPU. Examples range from object detection and pose estimation to instance segmentation and image denoising — all aimed at getting you started on edge AI quickly.

Why this repo exists:
- Provide simple examples you can adapt to your projects.
- Offer community project seeds you can fork and extend.

Quick links
- Sixfab Website: https://sixfab.com
- Sixfab Docs: https://docs.sixfab.com
- DEEPX Website: https://deepx.ai
- Runtime & tools: https://github.com/DEEPX-AI/dx_rt
- Driver: https://github.com/DEEPX-AI/dx_rt_npu_linux_driver
- Full toolchain: https://github.com/DEEPX-AI/dx-all-suite

Repository layout (high level)
- `basic_examples/` — Ready-to-run pipelines (detection, pose, segmentation, denoising).
- `community_projects/` — Community contributions and templates.
- `docs/` — Guides for installation and usage.
- `auto-install.sh` — helper scripts to prepare an environment.

Get started (fast)

1. Clone this repository:

```bash
git clone https://github.com/sixfab/deepx-rpi5-examples.git
cd deepx-rpi5-examples
```

2. Prepare the environment:

```bash
# Run the automated installer which also creates a virtual environment named `deepx`.
./auto-install.sh
# Activate the created environment:
source /home/$USER/deepx/bin/activate
```

3. Run a demo (example: object detection):

```bash
python basic_examples/object-detection/yolo-object-detection.py --rpicam
```

Notes on inputs
- For Raspberry Pi camera: pass `--rpicam` to examples that accept `--rpicam`.
- For USB webcams: most demos accept `--usbcam`.

Examples included
- Detection (light and full versions — the full one includes tracker and multi-resolution support).
- Pose estimation (realtime pose keypoints).
- Instance segmentation (masks + boxes).
- Image denoising.

Each example has usage notes and optional arguments in `docs/basic-examples.md`.

Community and contribution
We welcome community improvements. If you have a demo, model optimization tips, or a pipeline to share, add it under `community_projects/` and update `community_projects/community_projects.md` with usage notes.

Short disclaimer
These examples are provided for demonstration purposes. They’re intended to help you experiment with DEEPX NPU features on Raspberry Pi 5, not as production-ready systems. If something misbehaves, please open an issue in the repository.
