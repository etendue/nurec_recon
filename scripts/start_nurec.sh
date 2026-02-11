#!/bin/bash
# SPDX-FileCopyrightText: © 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

# Start NuRec Container using chroot (for Kubernetes/JupyterHub environments)
#
# Usage:
#   ./start_nurec.sh /path/to/scene.usdz [port]
#
# Environment Variables:
#   NUREC_ROOTFS      - Path to extracted rootfs (required, or set NUREC_IMAGE_TAR)
#   NUREC_IMAGE_TAR   - Path to exported image tar file (will extract to WORK_DIR/nurec-rootfs)
#   WORK_DIR          - Base directory for rootfs (default: /root/work/yusun)
#   CUDA_VISIBLE_DEVICES - GPU to use (default: 0)
#
# Example:
#   export NUREC_ROOTFS=/root/work/yusun/nurec-rootfs
#   ./start_nurec.sh /path/to/scene.usdz 46435

set -e

# Default work directory
WORK_DIR="${WORK_DIR:-/root/work/yusun}"

# Check arguments
if [ -z "$1" ]; then
    echo "Usage: $0 <usdz_path> [port]"
    echo ""
    echo "Arguments:"
    echo "  usdz_path: Path to the USDZ scene file"
    echo "  port:      Port for gRPC server (default: 46435)"
    echo ""
    echo "Environment Variables:"
    echo "  NUREC_ROOTFS      - Path to extracted rootfs"
    echo "  NUREC_IMAGE_TAR   - Path to image tar file (auto-extracts)"
    echo "  WORK_DIR          - Base directory (default: /root/work/yusun)"
    exit 1
fi

USDZ_PATH="$1"
PORT="${2:-46435}"

# Check if USDZ file exists
if [ ! -f "$USDZ_PATH" ]; then
    echo "Error: USDZ file not found: $USDZ_PATH"
    exit 1
fi

# Get absolute path
USDZ_PATH=$(realpath "$USDZ_PATH")
USDZ_DIR=$(dirname "$USDZ_PATH")

# Set GPU to use
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"

# Determine rootfs path
if [ -n "$NUREC_ROOTFS" ] && [ -d "$NUREC_ROOTFS" ]; then
    ROOTFS="$NUREC_ROOTFS"
else
    ROOTFS="$WORK_DIR/nurec-rootfs"
fi

echo "========================================"
echo "NuRec gRPC Server (chroot mode)"
echo "========================================"
echo "USDZ:    $USDZ_PATH"
echo "Port:    $PORT"
echo "GPU:     $GPU_ID"
echo "Rootfs:  $ROOTFS"
echo "========================================"

#######################################
# Extract tar file if rootfs doesn't exist
#######################################
extract_tar_if_needed() {
    if [ -d "$ROOTFS/app" ]; then
        echo "Rootfs already exists."
        return
    fi

    if [ -z "$NUREC_IMAGE_TAR" ] || [ ! -f "$NUREC_IMAGE_TAR" ]; then
        echo "Error: Rootfs not found and no tar file specified."
        echo ""
        echo "Please either:"
        echo "  1. Set NUREC_ROOTFS to existing rootfs directory"
        echo "  2. Set NUREC_IMAGE_TAR to image tar file"
        exit 1
    fi

    echo "Extracting tar file: $NUREC_IMAGE_TAR"
    mkdir -p "$ROOTFS"
    cd "$ROOTFS"

    # Extract to temp dir first
    TEMP_DIR=$(mktemp -d)
    tar -xf "$NUREC_IMAGE_TAR" -C "$TEMP_DIR"

    # Check format and extract layers
    if [ -f "$TEMP_DIR/manifest.json" ]; then
        echo "Processing Docker save format..."
        LAYERS=$(python3 -c "import json; m=json.load(open('$TEMP_DIR/manifest.json')); [print(l) for l in m[0]['Layers']]")
        for layer in $LAYERS; do
            echo "  Extracting: $layer"
            tar -xf "$TEMP_DIR/$layer" -C "$ROOTFS" 2>/dev/null || true
        done
    elif [ -d "$TEMP_DIR/blobs/sha256" ]; then
        echo "Processing OCI format..."
        INDEX_FILE="$TEMP_DIR/index.json"
        MANIFEST_DIGEST=$(python3 -c "import json; print(json.load(open('$INDEX_FILE'))['manifests'][0]['digest'].split(':')[1])")
        LAYERS=$(python3 -c "import json; [print(l['digest'].split(':')[1]) for l in json.load(open('$TEMP_DIR/blobs/sha256/$MANIFEST_DIGEST'))['layers']]")
        for layer in $LAYERS; do
            echo "  Extracting: ${layer:0:12}..."
            tar -xzf "$TEMP_DIR/blobs/sha256/$layer" -C "$ROOTFS" 2>/dev/null || \
            tar -xf "$TEMP_DIR/blobs/sha256/$layer" -C "$ROOTFS" 2>/dev/null || true
        done
    fi

    rm -rf "$TEMP_DIR"
    echo "Extraction complete."
}

#######################################
# Setup chroot environment
#######################################
setup_chroot() {
    echo "Setting up chroot environment..."

    # Create device nodes
    mkdir -p "$ROOTFS/dev"
    for dev in null zero random urandom; do
        cp -a "/dev/$dev" "$ROOTFS/dev/" 2>/dev/null || true
    done

    # Copy NVIDIA device nodes
    for dev in nvidia0 nvidia1 nvidia2 nvidia3 nvidia4 nvidia5 nvidia6 nvidia7 \
               nvidiactl nvidia-uvm nvidia-uvm-tools; do
        [ -e "/dev/$dev" ] && cp -a "/dev/$dev" "$ROOTFS/dev/" 2>/dev/null || true
    done

    # Copy NVIDIA libraries
    mkdir -p "$ROOTFS/usr/lib/x86_64-linux-gnu"
    cp /usr/bin/nvidia-smi "$ROOTFS/usr/bin/" 2>/dev/null || true
    cp -a /usr/lib/x86_64-linux-gnu/libnvidia* "$ROOTFS/usr/lib/x86_64-linux-gnu/" 2>/dev/null || true
    cp -a /usr/lib/x86_64-linux-gnu/libcuda* "$ROOTFS/usr/lib/x86_64-linux-gnu/" 2>/dev/null || true
    cp -a /usr/lib/x86_64-linux-gnu/libnvrtc* "$ROOTFS/usr/lib/x86_64-linux-gnu/" 2>/dev/null || true

    # Copy DNS config
    cp /etc/resolv.conf "$ROOTFS/etc/" 2>/dev/null || true
}

#######################################
# Mount directories
#######################################
do_mounts() {
    echo "Mounting directories..."

    # Mount /proc
    mkdir -p "$ROOTFS/proc"
    mountpoint -q "$ROOTFS/proc" || mount -t proc proc "$ROOTFS/proc" 2>/dev/null || true

    # Mount /sys
    mkdir -p "$ROOTFS/sys"
    mountpoint -q "$ROOTFS/sys" || mount --bind /sys "$ROOTFS/sys" 2>/dev/null || true

    # Mount /tmp
    mkdir -p "$ROOTFS/tmp"
    chmod 1777 "$ROOTFS/tmp"

    # Mount USDZ directory
    mkdir -p "$ROOTFS$USDZ_DIR"
    mountpoint -q "$ROOTFS$USDZ_DIR" || mount --bind "$USDZ_DIR" "$ROOTFS$USDZ_DIR" 2>/dev/null || {
        echo "Warning: Could not bind mount, copying USDZ file..."
        cp "$USDZ_PATH" "$ROOTFS$USDZ_DIR/" || true
    }
}

#######################################
# Cleanup mounts
#######################################
cleanup() {
    echo ""
    echo "Cleaning up..."
    umount "$ROOTFS$USDZ_DIR" 2>/dev/null || true
    umount "$ROOTFS/sys" 2>/dev/null || true
    umount "$ROOTFS/proc" 2>/dev/null || true
    echo "Done."
}

#######################################
# Run NuRec server
#######################################
run_server() {
    # Find entrypoint
    ENTRYPOINT="scripts/pycena/runtime/entrypoint_3_11.sh"

    echo ""
    echo "Starting NuRec server..."
    echo "  Entrypoint: $ENTRYPOINT"
    echo "  USDZ: $USDZ_PATH"
    echo "  Port: $PORT"
    echo ""
    echo "Press Ctrl+C to stop."
    echo ""

    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    
    # The entrypoint script uses relative paths (scripts/pycena/...),
    # so we need to run it from the correct working directory
    chroot "$ROOTFS" /bin/bash -c "
        cd /app/pycena_run.runfiles/nre_repo && 
        $ENTRYPOINT \
            --artifact-glob '$USDZ_PATH' \
            --port='$PORT' \
            --host=0.0.0.0 \
            --test-scenes-are-valid
    "
}

#######################################
# Main
#######################################
trap cleanup EXIT INT TERM

extract_tar_if_needed
setup_chroot
do_mounts
run_server
