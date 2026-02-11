#!/bin/bash
# SPDX-FileCopyrightText: © 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

# Start NuRec Container (Docker only)
#
# Usage:
#   ./start_nurec.sh /path/to/scene.usdz [port]
#
# Environment Variables:
#   NUREC_IMAGE            - Docker image (default: carlasimulator/nvidia-nurec-grpc:0.2.0)
#   CUDA_VISIBLE_DEVICES   - GPU id to use inside container (default: 0)
#   NUREC_CONTAINER_NAME   - Optional fixed container name

set -euo pipefail

if [ -z "${1:-}" ]; then
    echo "Usage: $0 <usdz_path> [port]"
    echo ""
    echo "Arguments:"
    echo "  usdz_path: Path to the USDZ scene file"
    echo "  port:      Port for gRPC server (default: 46435)"
    echo ""
    echo "Environment Variables:"
    echo "  NUREC_IMAGE            - Docker image to run"
    echo "  CUDA_VISIBLE_DEVICES   - GPU id to use (default: 0)"
    echo "  NUREC_CONTAINER_NAME   - Optional container name"
    exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "Error: docker command not found."
    exit 1
fi

USDZ_PATH="$1"
PORT="${2:-46435}"
NUREC_IMAGE="${NUREC_IMAGE:-carlasimulator/nvidia-nurec-grpc:0.2.0}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
CONTAINER_NAME="${NUREC_CONTAINER_NAME:-nurec-grpc-${PORT}}"

if [ ! -f "$USDZ_PATH" ]; then
    echo "Error: USDZ file not found: $USDZ_PATH"
    exit 1
fi

USDZ_PATH="$(realpath "$USDZ_PATH")"
USDZ_DIR="$(dirname "$USDZ_PATH")"

echo "========================================"
echo "NuRec gRPC Server (docker mode)"
echo "========================================"
echo "USDZ:       $USDZ_PATH"
echo "Port:       $PORT"
echo "GPU:        $GPU_ID"
echo "Image:      $NUREC_IMAGE"
echo "Container:  $CONTAINER_NAME"
echo "========================================"

echo ""
echo "Pulling image..."
docker pull "$NUREC_IMAGE"

EXISTING_CONTAINER_ID="$(docker ps -aq --filter "name=^/${CONTAINER_NAME}$")"
if [ -n "$EXISTING_CONTAINER_ID" ]; then
    echo "Removing existing container: $CONTAINER_NAME"
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
fi

cleanup() {
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo ""
echo "Starting NuRec server in Docker..."
echo "Press Ctrl+C to stop."
echo ""

docker run \
    --rm \
    --name "$CONTAINER_NAME" \
    --gpus all \
    --ipc=host \
    -e CUDA_VISIBLE_DEVICES="$GPU_ID" \
    -p "${PORT}:${PORT}" \
    -v "${USDZ_DIR}:${USDZ_DIR}:ro" \
    "$NUREC_IMAGE" \
    --artifact-glob "$USDZ_PATH" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --test-scenes-are-valid
