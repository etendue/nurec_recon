#!/bin/bash
# SPDX-FileCopyrightText: © 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

# Start all services for NuRec Web Viewer
#
# Usage:
#   ./start_all.sh /path/to/scene.usdz
#
# This script will:
# 1. Start NuRec container in background
# 2. Wait for it to be ready
# 3. Start the backend server
# 4. Start the frontend dev server

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <usdz_path>"
    exit 1
fi

USDZ_PATH="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEBAPP_DIR="$(dirname "$SCRIPT_DIR")"

echo "========================================"
echo "NuRec Web Viewer - Full Stack Startup"
echo "========================================"

# Start NuRec container in background
echo ""
echo "[1/3] Starting NuRec Container..."
echo "-------------------------------------"
"$SCRIPT_DIR/start_nurec.sh" "$USDZ_PATH" 46435 &
NUREC_PID=$!

# Wait for container to be ready
echo "Waiting for NuRec to initialize (this may take 1-2 minutes)..."
sleep 30

# Start backend
echo ""
echo "[2/3] Starting Backend Server..."
echo "-------------------------------------"
cd "$WEBAPP_DIR/backend"
pip install -r requirements.txt -q
PYTHONPATH="$(dirname "$WEBAPP_DIR"):$PYTHONPATH" uvicorn main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend
sleep 5

# Start frontend
echo ""
echo "[3/3] Starting Frontend Dev Server..."
echo "-------------------------------------"
cd "$WEBAPP_DIR/frontend"
npm install
npm run dev &
FRONTEND_PID=$!

echo ""
echo "========================================"
echo "All services started!"
echo "========================================"
echo "Frontend: http://localhost:3000"
echo "Backend:  http://localhost:8000"
echo "NuRec:    localhost:46435"
echo ""
echo "Press Ctrl+C to stop all services"
echo "========================================"

# Wait for any process to exit
wait $NUREC_PID $BACKEND_PID $FRONTEND_PID

# Cleanup
kill $NUREC_PID $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
