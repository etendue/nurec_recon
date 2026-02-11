# SPDX-FileCopyrightText: © 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import base64
import importlib
import logging
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from typing import Dict, Optional

import grpc
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from scipy.spatial.transform import Rotation, Slerp

from models import (
    CameraInfo,
    LoadRequest,
    LoadResponse,
    PoseAtTime,
    RenderRequest,
    RenderResponse,
    ScenarioInfo,
    TimeRange,
    TrajectoryPoint,
    TrajectoryResponse,
)

# Load protobuf modules generated from local ./proto at runtime.
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(BACKEND_DIR)
PROTO_SRC_DIR = os.path.join(REPO_DIR, "proto")
PROTO_GEN_DIR = os.path.join(BACKEND_DIR, ".cache", "proto_generated")


def _generate_proto_modules() -> None:
    os.makedirs(PROTO_GEN_DIR, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"-I{PROTO_SRC_DIR}",
        f"--python_out={PROTO_GEN_DIR}",
        f"--grpc_python_out={PROTO_GEN_DIR}",
        os.path.join(PROTO_SRC_DIR, "common.proto"),
        os.path.join(PROTO_SRC_DIR, "sensorsim.proto"),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _load_proto_modules() -> tuple[object, object, object]:
    if PROTO_GEN_DIR not in sys.path:
        sys.path.insert(0, PROTO_GEN_DIR)
    common_module = importlib.import_module("common_pb2")
    sensorsim_module = importlib.import_module("sensorsim_pb2")
    sensorsim_grpc_module = importlib.import_module("sensorsim_pb2_grpc")
    return common_module, sensorsim_module, sensorsim_grpc_module


try:
    common_pb2, sensorsim_pb2, sensorsim_pb2_grpc = _load_proto_modules()
    NUREC_AVAILABLE = True
except Exception:
    try:
        _generate_proto_modules()
        common_pb2, sensorsim_pb2, sensorsim_pb2_grpc = _load_proto_modules()
        NUREC_AVAILABLE = True
    except Exception as e:
        logging.warning(f"NuRec gRPC protobuf modules not available: {e}")
        NUREC_AVAILABLE = False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class AppState:
    """Global application state."""

    def __init__(self) -> None:
        self.grpc_channel = None
        self.grpc_stub = None
        self.nurec_host: str = "localhost"
        self.nurec_port: int = 46435
        self.scene_id: Optional[str] = None
        self.available_cameras: Dict[str, object] = {}
        self.available_trajectories = []


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting NuRec Web Viewer Backend...")
    yield
    if app_state.grpc_channel:
        app_state.grpc_channel.close()
    logger.info("Shutdown complete")


app = FastAPI(
    title="NuRec Web Viewer API",
    description="REST API for NuRec Web Viewer",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _pose_to_matrix(pose_msg) -> np.ndarray:
    rotation = Rotation.from_quat(
        [pose_msg.quat.x, pose_msg.quat.y, pose_msg.quat.z, pose_msg.quat.w]
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation.as_matrix()
    matrix[:3, 3] = [pose_msg.vec.x, pose_msg.vec.y, pose_msg.vec.z]
    return matrix


def _matrix_to_pose(matrix: np.ndarray):
    quat = Rotation.from_matrix(matrix[:3, :3]).as_quat()  # x, y, z, w
    return common_pb2.Pose(
        vec=common_pb2.Vec3(
            x=float(matrix[0, 3]),
            y=float(matrix[1, 3]),
            z=float(matrix[2, 3]),
        ),
        quat=common_pb2.Quat(
            w=float(quat[3]),
            x=float(quat[0]),
            y=float(quat[1]),
            z=float(quat[2]),
        ),
    )


def _select_scene_id(usdz_path: str, scene_ids: list[str]) -> str:
    if not scene_ids:
        raise HTTPException(status_code=500, detail="No scenes available in NuRec service.")
    preferred = os.path.splitext(os.path.basename(usdz_path))[0]
    if preferred in scene_ids:
        return preferred
    return scene_ids[0]


def _get_primary_trajectory():
    if not app_state.available_trajectories:
        return None
    # Prefer trajectory 0 if present, otherwise first available.
    for trajectory in app_state.available_trajectories:
        if trajectory.trajectory_idx == 0:
            return trajectory.trajectory
    return app_state.available_trajectories[0].trajectory


def _trajectory_bounds(trajectory) -> tuple[int, int]:
    poses = list(trajectory.poses)
    if not poses:
        return 0, 0
    return int(poses[0].timestamp_us), int(poses[-1].timestamp_us)


def _interpolate_pose_at(trajectory, timestamp_us: int):
    poses = list(trajectory.poses)
    if not poses:
        return None

    if timestamp_us <= poses[0].timestamp_us:
        return poses[0].pose
    if timestamp_us >= poses[-1].timestamp_us:
        return poses[-1].pose

    for idx in range(len(poses) - 1):
        start = poses[idx]
        end = poses[idx + 1]
        if start.timestamp_us <= timestamp_us <= end.timestamp_us:
            t0 = float(start.timestamp_us)
            t1 = float(end.timestamp_us)
            if t1 == t0:
                return start.pose
            alpha = (timestamp_us - t0) / (t1 - t0)

            start_pos = np.array([start.pose.vec.x, start.pose.vec.y, start.pose.vec.z], dtype=np.float64)
            end_pos = np.array([end.pose.vec.x, end.pose.vec.y, end.pose.vec.z], dtype=np.float64)
            interp_pos = (1.0 - alpha) * start_pos + alpha * end_pos

            start_rot = Rotation.from_quat([start.pose.quat.x, start.pose.quat.y, start.pose.quat.z, start.pose.quat.w])
            end_rot = Rotation.from_quat([end.pose.quat.x, end.pose.quat.y, end.pose.quat.z, end.pose.quat.w])
            slerp = Slerp([0.0, 1.0], Rotation.from_quat([start_rot.as_quat(), end_rot.as_quat()]))
            interp_rot = slerp([alpha])[0].as_quat()  # x, y, z, w

            return common_pb2.Pose(
                vec=common_pb2.Vec3(x=float(interp_pos[0]), y=float(interp_pos[1]), z=float(interp_pos[2])),
                quat=common_pb2.Quat(
                    w=float(interp_rot[3]),
                    x=float(interp_rot[0]),
                    y=float(interp_rot[1]),
                    z=float(interp_rot[2]),
                ),
            )
    return poses[-1].pose


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "nurec_available": NUREC_AVAILABLE,
        "scenario_loaded": app_state.scene_id is not None,
        "grpc_connected": app_state.grpc_stub is not None,
    }


@app.post("/api/load", response_model=LoadResponse)
async def load_scenario(request: LoadRequest):
    if not NUREC_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail="NuRec protobuf modules not available. Generate stubs from proto first.",
        )

    if not os.path.exists(request.usdz_path):
        raise HTTPException(status_code=404, detail=f"USDZ file not found: {request.usdz_path}")

    try:
        app_state.nurec_host = request.nurec_host
        app_state.nurec_port = request.nurec_port
        app_state.grpc_channel = grpc.insecure_channel(
            f"{request.nurec_host}:{request.nurec_port}",
            options=[
                ("grpc.max_send_message_length", 100 * 1024 * 1024),
                ("grpc.max_receive_message_length", 100 * 1024 * 1024),
            ],
        )
        app_state.grpc_stub = sensorsim_pb2_grpc.SensorsimServiceStub(app_state.grpc_channel)

        scenes_response = app_state.grpc_stub.get_available_scenes(
            common_pb2.Empty(),
            timeout=30.0,
        )
        scene_ids = list(scenes_response.scene_ids)
        app_state.scene_id = _select_scene_id(request.usdz_path, scene_ids)

        cameras_response = app_state.grpc_stub.get_available_cameras(
            sensorsim_pb2.AvailableCamerasRequest(scene_id=app_state.scene_id),
            timeout=120.0,
        )
        app_state.available_cameras = {
            cam.logical_id: cam for cam in cameras_response.available_cameras
        }

        try:
            trajectories_response = app_state.grpc_stub.get_available_trajectories(
                sensorsim_pb2.AvailableTrajectoriesRequest(scene_id=app_state.scene_id),
                timeout=20.0,
            )
            app_state.available_trajectories = list(trajectories_response.available_trajectories)
        except Exception as trajectory_error:
            logger.warning(f"Trajectory metadata unavailable: {trajectory_error}")
            app_state.available_trajectories = []

        return LoadResponse(status="loaded", sequence_id=app_state.scene_id)
    except Exception as e:
        logger.error(f"Failed to load scenario: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/scenario", response_model=ScenarioInfo)
async def get_scenario_info():
    if app_state.scene_id is None:
        raise HTTPException(status_code=400, detail="No scenario loaded. Call /api/load first.")

    trajectory = _get_primary_trajectory()
    start_us, end_us = _trajectory_bounds(trajectory) if trajectory else (0, 0)
    time_range = TimeRange(
        start_us=start_us,
        end_us=end_us,
        duration_seconds=(end_us - start_us) / 1_000_000 if end_us > start_us else 0.0,
    )

    cameras = []
    for logical_id, cam in app_state.available_cameras.items():
        camera_matrix = _pose_to_matrix(cam.rig_to_camera)
        intrinsics = cam.intrinsics
        cameras.append(
            CameraInfo(
                logical_name=logical_id,
                resolution=[int(intrinsics.resolution_w), int(intrinsics.resolution_h)],
                T_sensor_rig=camera_matrix.tolist(),
            )
        )

    return ScenarioInfo(sequence_id=app_state.scene_id, time_range=time_range, cameras=cameras)


@app.get("/api/trajectory", response_model=TrajectoryResponse)
async def get_trajectory(sample_interval_us: int = Query(default=100000, ge=10000)):
    trajectory = _get_primary_trajectory()
    if trajectory is None:
        raise HTTPException(status_code=400, detail="No trajectory data available.")

    points = []
    last_ts = None
    for pose_at_time in trajectory.poses:
        ts = int(pose_at_time.timestamp_us)
        if last_ts is not None and ts - last_ts < sample_interval_us:
            continue
        last_ts = ts
        pose = pose_at_time.pose
        points.append(
            TrajectoryPoint(
                timestamp_us=ts,
                position=[pose.vec.x, pose.vec.y, pose.vec.z],
                quaternion=[pose.quat.x, pose.quat.y, pose.quat.z, pose.quat.w],
            )
        )
    return TrajectoryResponse(trajectory=points)


@app.get("/api/pose", response_model=PoseAtTime)
async def get_pose_at_time(timestamp_us: int = Query(...)):
    trajectory = _get_primary_trajectory()
    if trajectory is None:
        raise HTTPException(status_code=400, detail="No trajectory data available.")
    pose = _interpolate_pose_at(trajectory, timestamp_us)
    if pose is None:
        raise HTTPException(status_code=400, detail="Timestamp out of range")
    return PoseAtTime(
        timestamp_us=timestamp_us,
        position=[pose.vec.x, pose.vec.y, pose.vec.z],
        quaternion=[pose.quat.x, pose.quat.y, pose.quat.z, pose.quat.w],
    )


@app.post("/api/render", response_model=RenderResponse)
async def render_cameras(request: RenderRequest):
    if app_state.grpc_stub is None or app_state.scene_id is None:
        raise HTTPException(status_code=400, detail="No scenario loaded or NuRec not connected")

    trajectory = _get_primary_trajectory()
    if trajectory is None:
        raise HTTPException(status_code=400, detail="No trajectory data available.")

    rig_pose = _interpolate_pose_at(trajectory, request.timestamp_us)
    if rig_pose is None:
        raise HTTPException(status_code=400, detail="Timestamp out of range")

    rig_matrix = _pose_to_matrix(rig_pose)
    results = {}

    for camera_id in request.camera_ids:
        cam = app_state.available_cameras.get(camera_id)
        if cam is None:
            logger.warning(f"Camera {camera_id} not found in available cameras")
            continue

        camera_matrix = rig_matrix @ _pose_to_matrix(cam.rig_to_camera)
        sensor_pose = _matrix_to_pose(camera_matrix)

        camera_spec = sensorsim_pb2.CameraSpec()
        camera_spec.CopyFrom(cam.intrinsics)
        camera_spec.resolution_h = max(1, int(camera_spec.resolution_h * request.resolution_scale))
        camera_spec.resolution_w = max(1, int(camera_spec.resolution_w * request.resolution_scale))

        grpc_request = sensorsim_pb2.RGBRenderRequest(
            scene_id=app_state.scene_id,
            resolution_h=camera_spec.resolution_h,
            resolution_w=camera_spec.resolution_w,
            camera_intrinsics=camera_spec,
            frame_start_us=int(request.timestamp_us),
            frame_end_us=int(request.timestamp_us) + 1,
            sensor_pose=sensorsim_pb2.PosePair(start_pose=sensor_pose, end_pose=sensor_pose),
            dynamic_objects=[],
            image_format=sensorsim_pb2.ImageFormat.JPEG,
            image_quality=90.0,
        )

        try:
            response = app_state.grpc_stub.render_rgb(grpc_request, timeout=10.0)
            results[camera_id] = base64.b64encode(response.image_bytes).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to render camera {camera_id}: {e}")

    return RenderResponse(timestamp_us=request.timestamp_us, images=results)


@app.get("/api/render/{camera_id}")
async def render_single_camera(
    camera_id: str,
    timestamp_us: int = Query(...),
    scale: float = Query(default=0.25, ge=0.1, le=1.0),
):
    request = RenderRequest(timestamp_us=timestamp_us, camera_ids=[camera_id], resolution_scale=scale)
    response = await render_cameras(request)
    if camera_id not in response.images:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found or render failed")
    return Response(content=base64.b64decode(response.images[camera_id]), media_type="image/jpeg")


@app.get("/api/cameras")
async def get_available_cameras():
    if app_state.scene_id is None:
        raise HTTPException(status_code=400, detail="No scenario loaded")
    return {"cameras": list(app_state.available_cameras.keys())}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
