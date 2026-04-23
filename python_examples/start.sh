#!/usr/bin/env bash
# start.sh — One-shot setup and launcher for DeepX SDK Python Demos
#
# What this script does (in order):
#   1. Installs the sixfab-dx package (DeepX NPU runtime)
#   2. Activates the pre-built Python venv that ships with it
#   3. Installs Python dependencies from requirements.txt
#   4. Downloads models and videos (skips if already present)
#   5. Launches the demo menu
#
# Usage:
#   chmod +x start.sh
#   ./start.sh
#
# On subsequent runs everything except the launcher is skipped automatically.

set -euo pipefail

# ── Colors ────────────────────────────────────────────────────────────────────

GREEN="\033[0;32m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
BOLD="\033[1m"
DIM="\033[2m"
RESET="\033[0m"

info()  { echo -e "${GREEN}[✓]${RESET} $*"; }
step()  { echo -e "\n${BOLD}$*${RESET}"; }
warn()  { echo -e "${YELLOW}[!]${RESET} $*"; }
error() { echo -e "${RED}[✗] $*${RESET}"; exit 1; }
dim()   { echo -e "${DIM}    $*${RESET}"; }

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESOURCES_DIR="$(dirname "$SCRIPT_DIR")/resources"
MODELS_DIR="$RESOURCES_DIR/models"
VIDEOS_DIR="$RESOURCES_DIR/videos"
VENV_PATH="/opt/sixfab-dx/venv"
VENV_PYTHON="$VENV_PATH/bin/python3"
VENV_PIP="$VENV_PATH/bin/pip"
REQUIREMENTS="$SCRIPT_DIR/requirements.txt"
LAUNCHER="$SCRIPT_DIR/launcher.py"

export RESOURCES_DIR MODELS_DIR VIDEOS_DIR

# ── Banner ────────────────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}  DeepX SDK Python Demos — Setup & Launch${RESET}"
echo -e "${DIM}  ─────────────────────────────────────────${RESET}"
echo ""

# ── Step 1 — Install sixfab-dx ────────────────────────────────────────────────

step "Step 1/4 — DeepX SDK (sixfab-dx)"

if command -v dxrt-cli &>/dev/null; then
    info "sixfab-dx is already installed."
    dim "$(dxrt-cli -s 2>/dev/null | head -1 || echo 'run: dxrt-cli -s to verify')"
else
    warn "sixfab-dx not found. Installing now..."

    # Add GPG key
    dim "Adding GPG key..."
    wget -qO - https://sixfab.github.io/sixfab_dx/public.gpg \
        | sudo gpg --dearmor -o /usr/share/keyrings/sixfab-dx.gpg

    # Add APT repository
    dim "Adding APT repository..."
    echo "deb [signed-by=/usr/share/keyrings/sixfab-dx.gpg] https://sixfab.github.io/sixfab_dx trixie main" \
        | sudo tee /etc/apt/sources.list.d/sixfab-dx.list > /dev/null

    # Install package
    dim "Running apt install..."
    sudo apt update -qq && sudo apt install -y sixfab-dx

    info "sixfab-dx installed successfully."

    # Verify
    if command -v dxrt-cli &>/dev/null; then
        dim "$(dxrt-cli -s 2>/dev/null | head -1 || true)"
    else
        error "dxrt-cli not found after install. Check the output above for errors."
    fi
fi

# ── Step 2 — Verify venv ──────────────────────────────────────────────────────

step "Step 2/4 — Python environment"

if [[ ! -f "$VENV_PYTHON" ]]; then
    error "venv not found at $VENV_PATH\n    Try reinstalling sixfab-dx: sudo apt install --reinstall sixfab-dx"
fi

info "Using venv at $VENV_PATH"
dim "$("$VENV_PYTHON" --version)"

# ── Step 3 — Install Python dependencies ─────────────────────────────────────

step "Step 3/4 — Python dependencies"

if [[ ! -f "$REQUIREMENTS" ]]; then
    error "requirements.txt not found at $REQUIREMENTS"
fi

dim "Installing from requirements.txt..."
"$VENV_PIP" install -q -r "$REQUIREMENTS"
info "Python dependencies are up to date."

# ── Step 4 — Launch ───────────────────────────────────────────────────────────

step "Step 4/4 — Launching demo menu"

if [[ ! -f "$LAUNCHER" ]]; then
    error "launcher.py not found at $LAUNCHER"
fi

info "Starting launcher...\n"

exec "$VENV_PYTHON" "$LAUNCHER"
