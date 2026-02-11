# NuRec Web Viewer

Web-based viewer for NuRec neural reconstruction scenarios. Design is consistent with the PhysicalAI-AV Camera Replay webapp.

## Features

- **Multi-Camera Display**: View up to 3 cameras simultaneously with draggable windows
- **Layout Presets**: Grid, Surround (vehicle perspective), Stack layouts
- **Playback Controls**: Play/Pause, ±1s/±5s step, timeline scrubbing
- **Speed Control**: 0.5x, 1x, 2x playback speed
- **Quality Control**: Low/Medium/High rendering quality
- **Camera Selection**: Choose which cameras to display
- **Status Bar**: Connection status and renderer info

## Architecture

```
Frontend (React)  -->  Backend (FastAPI)  -->  NuRec Container (gRPC)
   :3000                  :8000                     :46435
```

## Prerequisites

1. **NuRec Container Image**: Either:
   - Docker image: `NUREC_IMAGE=carlasimulator/nvidia-nurec-grpc:0.2.0`
2. **USDZ Scene File**: A valid NuRec scene file (download from HuggingFace)
3. **GPU**: NVIDIA GPU with CUDA support
4. **Node.js**: v18+ for frontend
5. **Python**: 3.10+ for backend

## Quick Start

### 1. Download USDZ Scene

```bash
# Login first (token must have gated dataset read access)
hf auth login

# Download one USDZ file to current directory
hf download nvidia/PhysicalAI-Autonomous-Vehicles-NuRec \
  "sample_set/25.07_release/Batch0001/026d6a39-bd8f-4175-bc61-fe50ed0403a3/026d6a39-bd8f-4175-bc61-fe50ed0403a3.usdz" \
  --repo-type dataset \
  --local-dir .
```

### 2. Start NuRec Container

```bash
export NUREC_IMAGE=carlasimulator/nvidia-nurec-grpc:0.2.0
./scripts/start_nurec.sh /path/to/scene.usdz 46435
```

Wait for the message "successfully loaded scene" or "serving on localhost:46435".

### 3. Start Backend

```bash
cd backend
pip install -r requirements.txt
./run.sh
```

Backend will be available at http://localhost:8000

### 4. Start Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend will be available at http://localhost:3000

### 5. Load Scene

1. Open http://localhost:3000 in your browser
2. Enter the USDZ file path
3. Click "Load"
4. Select cameras and use playback controls