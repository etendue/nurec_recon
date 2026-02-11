# SPDX-FileCopyrightText: © 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""
NuRec Web Viewer Backend - FastAPI Application

This module provides a REST API for the NuRec Web Viewer frontend.
It handles scenario loading, trajectory interpolation, and image rendering
by proxying requests to the NuRec gRPC service.

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import base64
import asyncio
import logging
from typing import Optional, Dict
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

# Add parent directory to path to import nurec modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    LoadRequest,
    LoadResponse,
    ScenarioInfo,
    TimeRange,
    CameraInfo,
    TrajectoryResponse,
    TrajectoryPoint,
    PoseAtTime,
    RenderRequest,
    RenderResponse,
)

# Import nurec modules (these should be available from parent directory)
try:
    from scenario import Scenario
    from nre.grpc.protos.sensorsim_pb2_grpc import SensorsimServiceStub
    from nre.grpc.protos.sensorsim_pb2 import (
        RGBRenderRequest,
        AvailableCamerasRequest,
        CameraSpec,
        PosePair,
        ImageFormat,
        FthetaCameraParam,
        ShutterType,
    )
    from nre.grpc.protos.common_pb2 import Empty as EmptyRequest, Pose, Vec3, Quat
    import grpc
    NUREC_AVAILABLE = True
except ImportError as e:
    logging.warning(f"NuRec modules not available: {e}")
    NUREC_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============ Global State ============

class AppState:
    """Global application state"""
    def __init__(self):
        self.scenario: Optional[Scenario] = None
        self.grpc_channel = None
        self.grpc_stub = None
        self.available_cameras: Dict[str, CameraSpec] = {}
        self.nurec_host: str = "localhost"
        self.nurec_port: int = 46435

app_state = AppState()


# ============ Lifespan ============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    logger.info("Starting NuRec Web Viewer Backend...")
    yield
    # Cleanup
    if app_state.grpc_channel:
        app_state.grpc_channel.close()
    logger.info("Shutdown complete")


# ============ FastAPI App ============

app = FastAPI(
    title="NuRec Web Viewer API",
    description="REST API for NuRec Web Viewer",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ Helper Functions ============

def matrix_to_pose(matrix: np.ndarray) -> Pose:
    """Convert 4x4 transformation matrix to gRPC Pose message"""
    from scipy.spatial.transform import Rotation
    
    # Extract translation
    translation = matrix[:3, 3]
    
    # Extract rotation as quaternion
    rotation = Rotation.from_matrix(matrix[:3, :3])
    quat = rotation.as_quat()  # [x, y, z, w]
    
    return Pose(
        vec=Vec3(x=float(translation[0]), y=float(translation[1]), z=float(translation[2])),
        quat=Quat(w=float(quat[3]), x=float(quat[0]), y=float(quat[1]), z=float(quat[2]))
    )


def build_camera_spec(camera_id: str, cal, scale: float = 1.0) -> CameraSpec:
    """Build CameraSpec from camera calibration"""
    params = cal.camera_model.parameters
    
    resolution_h = int(params.resolution[1] * scale)
    resolution_w = int(params.resolution[0] * scale)
    
    # Build FthetaCameraParam
    ftheta_param = FthetaCameraParam(
        principal_point_x=params.principal_point[0] * scale,
        principal_point_y=params.principal_point[1] * scale,
        reference_poly=1 if params.reference_poly == "PIXELDIST_TO_ANGLE" else 2,
        pixeldist_to_angle_poly=params.pixeldist_to_angle_poly,
        angle_to_pixeldist_poly=params.angle_to_pixeldist_poly,
        max_angle=params.max_angle,
    )
    
    return CameraSpec(
        ftheta_param=ftheta_param,
        logical_id=camera_id,
        trajectory_idx=0,
        resolution_h=resolution_h,
        resolution_w=resolution_w,
        shutter_type=ShutterType.GLOBAL,
    )


# ============ API Endpoints ============

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "nurec_available": NUREC_AVAILABLE,
        "scenario_loaded": app_state.scenario is not None,
        "grpc_connected": app_state.grpc_stub is not None
    }


@app.post("/api/load", response_model=LoadResponse)
async def load_scenario(request: LoadRequest):
    """
    Load a USDZ scenario and connect to NuRec gRPC service.
    
    The NuRec container must already be running with the same USDZ file loaded.
    """
    if not NUREC_AVAILABLE:
        raise HTTPException(
            status_code=500, 
            detail="NuRec modules not available. Make sure nurec package is in PYTHONPATH."
        )
    
    if not os.path.exists(request.usdz_path):
        raise HTTPException(status_code=404, detail=f"USDZ file not found: {request.usdz_path}")
    
    try:
        # Load scenario
        logger.info(f"Loading scenario from {request.usdz_path}")
        app_state.scenario = Scenario(request.usdz_path)
        
        # Connect to gRPC
        logger.info(f"Connecting to NuRec at {request.nurec_host}:{request.nurec_port}")
        app_state.nurec_host = request.nurec_host
        app_state.nurec_port = request.nurec_port
        
        app_state.grpc_channel = grpc.insecure_channel(
            f"{request.nurec_host}:{request.nurec_port}",
            options=[
                ("grpc.max_send_message_length", 100 * 1024 * 1024),
                ("grpc.max_receive_message_length", 100 * 1024 * 1024),
            ]
        )
        app_state.grpc_stub = SensorsimServiceStub(app_state.grpc_channel)
        
        # Get available cameras from NuRec
        scene_id = app_state.scenario.metadata["sequence_id"]
        cameras_response = app_state.grpc_stub.get_available_cameras(
            AvailableCamerasRequest(scene_id=scene_id)
        )
        
        app_state.available_cameras = {}
        for cam in cameras_response.available_cameras:
            app_state.available_cameras[cam.logical_id] = cam.intrinsics
        
        logger.info(f"Loaded scenario with {len(app_state.available_cameras)} cameras")
        
        return LoadResponse(
            status="loaded",
            sequence_id=scene_id
        )
        
    except Exception as e:
        logger.error(f"Failed to load scenario: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/scenario", response_model=ScenarioInfo)
async def get_scenario_info():
    """Get scenario information including time range and available cameras"""
    if app_state.scenario is None:
        raise HTTPException(status_code=400, detail="No scenario loaded. Call /api/load first.")
    
    metadata = app_state.scenario.metadata
    
    start_us = metadata["pose-range"]["start-timestamp_us"]
    end_us = metadata["pose-range"]["end-timestamp_us"]
    
    time_range = TimeRange(
        start_us=start_us,
        end_us=end_us,
        duration_seconds=(end_us - start_us) / 1_000_000
    )
    
    cameras = []
    for cal in app_state.scenario.camera_calibrations.values():
        cameras.append(CameraInfo(
            logical_name=cal.logical_sensor_name,
            resolution=cal.camera_model.parameters.resolution,
            T_sensor_rig=cal.T_sensor_rig
        ))
    
    return ScenarioInfo(
        sequence_id=metadata["sequence_id"],
        time_range=time_range,
        cameras=cameras
    )


@app.get("/api/trajectory", response_model=TrajectoryResponse)
async def get_trajectory(sample_interval_us: int = Query(default=100000, ge=10000)):
    """
    Get the ego trajectory sampled at regular intervals.
    
    Args:
        sample_interval_us: Sampling interval in microseconds (default: 100ms)
    """
    if app_state.scenario is None:
        raise HTTPException(status_code=400, detail="No scenario loaded")
    
    metadata = app_state.scenario.metadata
    start_us = metadata["pose-range"]["start-timestamp_us"]
    end_us = metadata["pose-range"]["end-timestamp_us"]
    
    trajectory = []
    for t in range(start_us, end_us, sample_interval_us):
        pose = app_state.scenario.ego_poses.interpolate_pose_xyzquat(t)
        if pose:
            trajectory.append(TrajectoryPoint(
                timestamp_us=t,
                position=pose[:3],
                quaternion=pose[3:7]
            ))
    
    return TrajectoryResponse(trajectory=trajectory)


@app.get("/api/pose", response_model=PoseAtTime)
async def get_pose_at_time(timestamp_us: int = Query(...)):
    """Get the ego pose at a specific timestamp"""
    if app_state.scenario is None:
        raise HTTPException(status_code=400, detail="No scenario loaded")
    
    pose = app_state.scenario.ego_poses.interpolate_pose_xyzquat(timestamp_us)
    if pose is None:
        raise HTTPException(status_code=400, detail="Timestamp out of range")
    
    return PoseAtTime(
        timestamp_us=timestamp_us,
        position=pose[:3],
        quaternion=pose[3:7]
    )


@app.post("/api/render", response_model=RenderResponse)
async def render_cameras(request: RenderRequest):
    """
    Render images from specified cameras at a given timestamp.
    
    Returns base64 encoded JPEG images for each requested camera.
    """
    if app_state.scenario is None or app_state.grpc_stub is None:
        raise HTTPException(status_code=400, detail="No scenario loaded or NuRec not connected")
    
    # Get ego pose at timestamp
    ego_pose_mat = app_state.scenario.ego_poses.interpolate_pose_matrix(request.timestamp_us)
    if ego_pose_mat is None:
        raise HTTPException(status_code=400, detail="Timestamp out of range")
    
    scene_id = app_state.scenario.metadata["sequence_id"]
    end_timestamp = app_state.scenario.metadata["pose-range"]["end-timestamp_us"]
    
    # Clamp timestamp to valid range
    timestamp = min(request.timestamp_us, end_timestamp - 1)
    
    results = {}
    
    for camera_id in request.camera_ids:
        # Find camera calibration
        cam_cal = None
        for cal in app_state.scenario.camera_calibrations.values():
            if cal.logical_sensor_name == camera_id:
                cam_cal = cal
                break
        
        if cam_cal is None:
            logger.warning(f"Camera {camera_id} not found in scenario")
            continue
        
        # Calculate camera pose in world coordinates
        T_sensor_rig = np.array(cam_cal.T_sensor_rig)
        camera_pose_world = ego_pose_mat @ T_sensor_rig
        
        # Build camera spec
        camera_spec = build_camera_spec(camera_id, cam_cal, request.resolution_scale)
        
        # Build gRPC request
        sensor_pose = matrix_to_pose(camera_pose_world)
        
        grpc_request = RGBRenderRequest(
            scene_id=scene_id,
            resolution_h=camera_spec.resolution_h,
            resolution_w=camera_spec.resolution_w,
            camera_intrinsics=camera_spec,
            frame_start_us=timestamp,
            frame_end_us=timestamp + 1,
            sensor_pose=PosePair(
                start_pose=sensor_pose,
                end_pose=sensor_pose
            ),
            dynamic_objects=[],
            image_format=ImageFormat.JPEG,
            image_quality=90
        )
        
        try:
            # Call NuRec render
            response = app_state.grpc_stub.render_rgb(grpc_request)
            
            # Encode as base64
            image_b64 = base64.b64encode(response.image_bytes).decode('utf-8')
            results[camera_id] = image_b64
            
        except Exception as e:
            logger.error(f"Failed to render camera {camera_id}: {e}")
            continue
    
    return RenderResponse(
        timestamp_us=request.timestamp_us,
        images=results
    )


@app.get("/api/render/{camera_id}")
async def render_single_camera(
    camera_id: str,
    timestamp_us: int = Query(...),
    scale: float = Query(default=0.25, ge=0.1, le=1.0)
):
    """
    Render a single camera and return the image directly as JPEG.
    
    This is useful for <img> tags that can directly reference this URL.
    """
    request = RenderRequest(
        timestamp_us=timestamp_us,
        camera_ids=[camera_id],
        resolution_scale=scale
    )
    
    response = await render_cameras(request)
    
    if camera_id not in response.images:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found or render failed")
    
    image_bytes = base64.b64decode(response.images[camera_id])
    return Response(content=image_bytes, media_type="image/jpeg")


@app.get("/api/cameras")
async def get_available_cameras():
    """Get list of available camera IDs"""
    if app_state.scenario is None:
        raise HTTPException(status_code=400, detail="No scenario loaded")
    
    cameras = [cal.logical_sensor_name for cal in app_state.scenario.camera_calibrations.values()]
    return {"cameras": cameras}


# ============ Main ============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
