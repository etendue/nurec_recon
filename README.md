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
   - Or tar file: `NUREC_IMAGE_TAR=/path/to/nvidia-nurec-grpc.tar`
2. **USDZ Scene File**: A valid NuRec scene file (download from HuggingFace)
3. **GPU**: NVIDIA GPU with CUDA support
4. **Node.js**: v18+ for frontend
5. **Python**: 3.10+ for backend

## Quick Start

### 1. Download USDZ Scene

```bash
# Set up environment
source ~/.bashrc  # HF_ENDPOINT should be set to https://hf-mirror.com

# Download a sample scene
uv run python scripts/download_nurec_usdz.py --sample

# Or download specific scenes by criteria
uv run python scripts/download_nurec_usdz.py --weather clear/cloudy --limit 5
```

### 2. Start NuRec Container

**Docker mode** (if Docker is available):
```bash
export NUREC_IMAGE=carlasimulator/nvidia-nurec-grpc:0.2.0
./scripts/start_nurec.sh /path/to/scene.usdz 46435
```

**Chroot mode** (for Kubernetes/JupyterHub without Docker):
```bash
# Option A: From tar file (if image is exported as tar)
export NUREC_IMAGE_TAR=/path/to/nvidia-nurec-grpc.tar
./scripts/start_nurec.sh /path/to/scene.usdz 46435

# Option B: Download via skopeo
export NUREC_IMAGE=carlasimulator/nvidia-nurec-grpc:0.2.0
export USE_CHROOT=1
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

## USDZ Download Script

The `scripts/download_nurec_usdz.py` script downloads NuRec scenes from HuggingFace:

```bash
# List available scenes
uv run python scripts/download_nurec_usdz.py --list

# Download a sample scene
uv run python scripts/download_nurec_usdz.py --sample

# Download by batch
uv run python scripts/download_nurec_usdz.py --batch Batch0002

# Download by scene UUID
uv run python scripts/download_nurec_usdz.py --scene abc123-def456

# Filter by labels (Batch0002+)
uv run python scripts/download_nurec_usdz.py --weather rain --limit 5
uv run python scripts/download_nurec_usdz.py --road-types urban --limit 10
```

## NuRec Container Modes

The `start_nurec.sh` script supports two modes:

### Docker Mode (default)
- Requires Docker daemon
- Set `NUREC_IMAGE` environment variable
- Uses GPU passthrough via `--gpus all`

### Chroot Mode (for restricted environments)
- For Kubernetes/JupyterHub containers without Docker
- Uses skopeo + chroot (per `PodmanRunOnJupyterContainerWithGPU.md`)
- Set either:
  - `NUREC_IMAGE_TAR`: Path to exported image tar file
  - `NUREC_ROOTFS`: Path to pre-extracted rootfs
  - Or force via `USE_CHROOT=1`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/load` | POST | Load a USDZ scenario |
| `/api/scenario` | GET | Get scenario info (time range, cameras) |
| `/api/trajectory` | GET | Get ego trajectory |
| `/api/pose` | GET | Get pose at timestamp |
| `/api/render` | POST | Render camera images |
| `/api/render/{camera_id}` | GET | Render single camera (returns JPEG) |

## Project Structure

```
webapp2/
├── DESIGN.md           # Detailed design document
├── README.md           # This file
├── backend/
│   ├── main.py         # FastAPI application
│   ├── models.py       # Pydantic models
│   ├── requirements.txt
│   └── run.sh          # Startup script
├── frontend/
│   ├── src/
│   │   ├── App.tsx     # Main React component
│   │   ├── api.ts      # API client
│   │   ├── types.ts    # TypeScript types
│   │   ├── index.css   # Styles (NVIDIA theme)
│   │   ├── components/
│   │   │   ├── CameraGrid.tsx      # Draggable camera display
│   │   │   ├── CameraSelector.tsx  # Camera selection
│   │   │   └── PlaybackControls.tsx # Playback + quality controls
│   │   └── hooks/
│   │       └── useNuRecPlayback.ts # Playback logic
│   ├── package.json
│   └── vite.config.ts
└── scripts/
    ├── start_nurec.sh  # Start NuRec container (Docker or chroot)
    └── start_all.sh    # Start all services
```

## Configuration

### Backend Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PYTHONPATH` | - | Must include nurec module directory |

### Frontend Configuration

Edit `vite.config.ts` to change:
- Frontend port (default: 3000)
- Backend proxy URL (default: localhost:8000)

### NuRec Container Environment Variables

| Variable | Description |
|----------|-------------|
| `NUREC_IMAGE` | Docker image name (e.g., carlasimulator/nvidia-nurec-grpc:0.2.0) |
| `NUREC_IMAGE_TAR` | Path to exported image tar file (for chroot mode) |
| `NUREC_ROOTFS` | Path to extracted rootfs (for chroot mode) |
| `WORK_DIR` | Base directory for image storage (default: /root/work/yusun) |
| `USE_CHROOT` | Force chroot mode if set to "1" |
| `CUDA_VISIBLE_DEVICES` | GPU to use (default: 0) |

## UI Features

### Draggable Camera Windows
- Drag camera windows by their header
- Resize windows using the corner handle
- Positions are saved to localStorage

### Layout Presets
- **Grid**: Evenly distributed grid layout
- **Surround**: Vehicle-perspective layout (front center, left/right sides)
- **Stack**: Vertical stack layout
- **Reset**: Clear saved positions

### Quality Control
- **Low**: Fast rendering, lower resolution (0.15x scale)
- **Medium**: Balanced (0.25x scale, default)
- **High**: Best quality, slower (0.5x scale)

## Troubleshooting

### "NuRec modules not available"

Make sure the nurec package is in PYTHONPATH:
```bash
export PYTHONPATH=/path/to/nurec:$PYTHONPATH
```

### "No scenario loaded"

Call `/api/load` first with the USDZ file path.

### Slow rendering

- Set Quality to "Low" for faster rendering
- Use fewer cameras
- Check GPU utilization: `nvidia-smi`

### Container won't start (Docker mode)

- Check `NUREC_IMAGE` is set correctly
- Verify GPU is available: `nvidia-smi`
- Check USDZ file path is correct

### Container won't start (Chroot mode)

- Check tar file exists: `ls -la $NUREC_IMAGE_TAR`
- Verify skopeo is installed: `which skopeo`
- Check rootfs extraction: `ls $WORK_DIR/nurec-rootfs/usr`
- Ensure NVIDIA devices exist: `ls /dev/nvidia*`

## Development

### Backend Development

```bash
cd backend
uvicorn main:app --reload --port 8000
```

### Frontend Development

```bash
cd frontend
npm run dev
```

## License

SPDX-License-Identifier: MIT
