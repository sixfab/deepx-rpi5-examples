# DEEPX Raspberry Pi 5 — Installation Guide

This guide covers hardware requirements and how to install the DEEPX NPU runtime on a Raspberry Pi 5.

## Table of Contents
- [Hardware Requirements](#hardware-requirements)
- [Operating System Setup](#operating-system-setup)
- [Installing the DEEPX Runtime](#installing-the-deepx-runtime)
- [Setting Up the Examples](#setting-up-the-examples)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)
- [Additional Resources](#additional-resources)

---

## Hardware Requirements

| Component | Details |
|-----------|---------|
| Raspberry Pi 5 | 8 GB RAM recommended |
| DEEPX NPU Module | Required for all inference |
| MicroSD Card | 32 GB or larger, Class 10 or better |
| Power Supply | Official RPi 5 power adapter recommended |
| Camera | Optional — RPi Camera Module or USB webcam |
| Cooling | Heatsink or fan recommended for sustained inference |

---

## Operating System Setup

Flash **Raspberry Pi OS 64-bit (Bookworm)** to your SD card using the [Raspberry Pi Imager](https://www.raspberrypi.com/software/).

After first boot, update the system and install build dependencies:

```bash
sudo apt update && sudo apt upgrade -y

sudo apt install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    curl
```

---

## Installing the DEEPX Runtime

The DEEPX runtime is distributed through the Sixfab APT repository. This installs the runtime, Python wheels, and a pre-configured virtual environment at `/opt/sixfab-dx/venv`.

**Step 1 — Add the Sixfab GPG key**

```bash
wget -qO - https://sixfab.github.io/sixfab_dx/public.gpg | sudo gpg --dearmor -o /usr/share/keyrings/sixfab-dx.gpg
```

**Step 2 — Add the APT repository**

```bash
echo "deb [signed-by=/usr/share/keyrings/sixfab-dx.gpg] https://sixfab.github.io/sixfab_dx trixie main" | sudo tee /etc/apt/sources.list.d/sixfab-dx.list
```

**Step 3 — Install the package**

```bash
sudo apt update && sudo apt install sixfab-dx
```

**Step 4 — Verify the NPU is recognized**

```bash
dxrt-cli -s
```

The command prints the NPU device status. If the device is listed, the runtime is ready.

**Step 5 — Activate the Python environment**

```bash
source /opt/sixfab-dx/venv/bin/activate
```

Add this line to your `~/.bashrc` or run it at the start of each session.

<details>
<summary>Updating sixfab-dx</summary>

```bash
sudo apt update && sudo apt upgrade sixfab-dx
```
</details>

<details>
<summary>Uninstalling sixfab-dx</summary>

```bash
sudo apt remove sixfab-dx
sudo rm /etc/apt/sources.list.d/sixfab-dx.list
sudo rm /usr/share/keyrings/sixfab-dx.gpg
sudo apt update
```
</details>

### Alternative: Install from source (DEEPX upstream)

If you need to build from source rather than use the Sixfab APT package:

```bash
# Clone and install the NPU kernel driver
git clone https://github.com/DEEPX-AI/dx_rt_npu_linux_driver
cd dx_rt_npu_linux_driver
./install.sh
cd ~

# Clone and build the runtime
git clone https://github.com/DEEPX-AI/dx_rt
cd dx_rt
./install.sh --all
./build.sh

# Create a virtual environment and install the Python package
python3 -m venv --system-site-packages ~/deepx
source ~/deepx/bin/activate
pip install . ~/dx_rt/python_package
```

### Alternative: Official DEEPX all-in-one suite

```bash
git clone --recurse-submodules https://github.com/DEEPX-AI/dx-all-suite
cd dx-all-suite
python3 -m venv --system-site-packages deepx
source deepx/bin/activate
./dx-runtime/install.sh --all
```

---

## Setting Up the Examples

Once `sixfab-dx` is installed and `dxrt-cli -s` confirms the NPU is working, run the setup script from the repository root:

```bash
git clone https://github.com/sixfab/deepx-rpi5-examples.git
cd deepx-rpi5-examples
./auto-install.sh
```

`auto-install.sh` does **not** install the DEEPX runtime — that step must be done first (see above). It performs the following:

1. Validates `dxrt-cli` is installed and the NPU is working
2. Downloads models, sample images, and videos from the GitHub release into `resources/`
3. Activates `/opt/sixfab-dx/venv` and installs Python dependencies
4. Builds the C++ examples via `cpp_examples/setup.sh`

After setup, activate the environment for each new session:

```bash
source /opt/sixfab-dx/venv/bin/activate
```

---

## Verification

Check that the runtime and examples are working:

```bash
# Confirm the kernel module is loaded
lsmod | grep dx

# Confirm the NPU is detected
dxrt-cli -s

# Run a Python example
source /opt/sixfab-dx/venv/bin/activate
bash python_examples/start.sh
```

---

## Troubleshooting

**NPU not detected by dxrt-cli**

```bash
# Check kernel module status
lsmod | grep dx
dmesg | grep -i dx

# Check the runtime daemon
sudo journalctl -u dxrtd
```

If the module is not loaded, reload it or check the driver installation:

```bash
sudo modprobe dx_npu
```

**Camera not found**

```bash
# List available cameras
ls /dev/video*
libcamera-hello --list-cameras

# Enable the camera interface if needed
sudo raspi-config  # → Interface Options → Camera → Enable
```

**auto-install.sh fails with "DEEPX NPU runtime is not installed"**

Install `sixfab-dx` using the APT method above before running `auto-install.sh`.

**Python ImportError: No module named 'dxrt'**

The Sixfab venv is not active. Run:

```bash
source /opt/sixfab-dx/venv/bin/activate
```

---

## Next Steps

After successful installation:

1. **Python demos** — launch the 28-demo TUI with `bash python_examples/start.sh`; see [python-examples.md](python-examples.md)
2. **C++ demos** — launch the 26-demo TUI with `bash cpp_examples/start.sh`; see [cpp-examples.md](cpp-examples.md)
3. **Community projects** — contribute in `community_projects/`

---

## Additional Resources

- [DEEPX Official Website](https://www.deepx.ai/)
- [dx_rt Repository](https://github.com/DEEPX-AI/dx_rt)
- [dx_rt_npu_linux_driver Repository](https://github.com/DEEPX-AI/dx_rt_npu_linux_driver)
- [dx-all-suite Repository](https://github.com/DEEPX-AI/dx-all-suite)
- [Sixfab Documentation](https://docs.sixfab.com/)

---

*If you encounter issues or have suggestions, please open an issue on [GitHub](https://github.com/sixfab/deepx-rpi5-examples).*
