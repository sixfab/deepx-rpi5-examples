#!/usr/bin/env bash
# download.sh — Download models and videos for DeepX SDK Python Demos

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GITHUB_REPO="sixfab/deepx-sdk-python-demos"   # GitHub user/repo
RELEASE_TAG="${RELEASE_TAG:-latest}"            # Override: RELEASE_TAG=v1.0.0 ./download.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/models"
VIDEOS_DIR="$SCRIPT_DIR/videos"
MODELS_ARCHIVE="models.tar.gz"
VIDEOS_ARCHIVE="videos.tar.gz"

# ---------------------------------------------------------------------------
# ANSI colors
# ---------------------------------------------------------------------------
if [[ -t 1 ]]; then
    C_GREEN=$'\033[0;32m'
    C_YELLOW=$'\033[0;33m'
    C_RED=$'\033[0;31m'
    C_DIM=$'\033[2;37m'
    C_RESET=$'\033[0m'
else
    C_GREEN=""
    C_YELLOW=""
    C_RED=""
    C_DIM=""
    C_RESET=""
fi

info()  { printf '%s[INFO]%s  %s\n'  "$C_GREEN"  "$C_RESET" "$*"; }
warn()  { printf '%s[WARN]%s  %s\n'  "$C_YELLOW" "$C_RESET" "$*"; }
error() { printf '%s[ERROR]%s %s\n'  "$C_RED"    "$C_RESET" "$*" >&2; }
dim()   { printf '%s%s%s\n'          "$C_DIM"    "$*"       "$C_RESET"; }

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
if [[ "$GITHUB_REPO" == "REPLACE_ME/REPLACE_ME" ]] || [[ -z "$GITHUB_REPO" ]]; then
    error "GITHUB_REPO is not configured. Edit this script and set GITHUB_REPO to '<user>/<repo>'."
    exit 1
fi

if ! command -v wget >/dev/null 2>&1; then
    error "wget is not installed. Install it with: sudo apt install wget"
    exit 1
fi

if ! command -v tar >/dev/null 2>&1; then
    error "tar is not installed. Install it with: sudo apt install tar"
    exit 1
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
build_url() {
    local filename="$1"
    if [[ "$RELEASE_TAG" == "latest" ]]; then
        printf 'https://github.com/%s/releases/latest/download/%s' "$GITHUB_REPO" "$filename"
    else
        printf 'https://github.com/%s/releases/download/%s/%s' "$GITHUB_REPO" "$RELEASE_TAG" "$filename"
    fi
}

is_dir_non_empty() {
    local dir="$1"
    [[ -d "$dir" ]] && [[ -n "$(ls -A "$dir" 2>/dev/null)" ]]
}

# Download + extract one archive into a target directory. Skips when the
# target already has content so that re-runs are cheap and non-destructive.
fetch_and_extract() {
    local label="$1"
    local archive="$2"
    local target_dir="$3"

    if is_dir_non_empty "$target_dir"; then
        warn "${label} directory already exists and is not empty — skipping."
        dim "Path : $target_dir"
        dim "To re-download, empty the directory first and run again."
        return 0
    fi

    mkdir -p "$target_dir"

    local url
    url="$(build_url "$archive")"
    local tmp_file
    tmp_file="$(mktemp --suffix=".tar.gz")"

    # Always remove the temp file, success or failure.
    trap 'rm -f "$tmp_file"' RETURN

    info "Downloading ${label}..."
    dim "URL : $url"
    dim "Into: $target_dir"

    if ! wget --show-progress --quiet -O "$tmp_file" "$url"; then
        error "Failed to download ${label} from:"
        dim "  $url"
        error "Check the URL, your internet connection, and the RELEASE_TAG."
        rm -f "$tmp_file"
        exit 1
    fi

    info "Extracting ${label}..."
    # Try --strip-components=1 first (archive has a top-level folder such as
    # models/ or videos/). Fall back to a plain extraction if the archive
    # dumps files directly.
    if ! tar -xzf "$tmp_file" -C "$target_dir" --strip-components=1 2>/dev/null \
        && ! tar -xzf "$tmp_file" -C "$target_dir"; then
        error "Failed to extract ${archive} into ${target_dir}."
        dim "The archive may be corrupted or in an unexpected format."
        rm -f "$tmp_file"
        exit 1
    fi

    rm -f "$tmp_file"
    local file_count
    file_count="$(find "$target_dir" -type f | wc -l)"
    info "${label} ready."
    dim "Files: $file_count"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
info "DeepX SDK Python Demos — asset downloader"
dim "Repo   : $GITHUB_REPO"
dim "Release: $RELEASE_TAG"
printf '\n'

fetch_and_extract "Models" "$MODELS_ARCHIVE" "$MODELS_DIR"
printf '\n'
fetch_and_extract "Videos" "$VIDEOS_ARCHIVE" "$VIDEOS_DIR"
printf '\n'

info "All done. You can now run the demos:"
dim "source /opt/sixfab-dx/venv/bin/activate"
dim "python launcher.py"
