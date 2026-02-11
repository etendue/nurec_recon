#!/bin/bash
# SPDX-FileCopyrightText: © 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

# NuRec Web Viewer Backend Startup Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NUREC_DIR="$(dirname "$SCRIPT_DIR")"

# Add nurec directory to PYTHONPATH
export PYTHONPATH="${NUREC_DIR}:${PYTHONPATH}"

echo "Starting NuRec Web Viewer Backend..."
echo "PYTHONPATH: ${PYTHONPATH}"

cd "$SCRIPT_DIR"

# Check if uvicorn is installed
if ! command -v uvicorn &> /dev/null; then
    echo "uvicorn not found. Installing dependencies..."
    pip install -r requirements.txt
fi

# Start the server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
