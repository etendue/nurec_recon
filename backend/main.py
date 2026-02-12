# SPDX-FileCopyrightText: © 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import base64
import importlib
import logging
import os
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import grpc
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from scipy.spatial.transform import Rotation

from models import (
    CameraInfo,
    PlaybackControlRequest,
    PlaybackStateResponse,
    PlaybackTickRequest,
    PlaybackTickResponse,
    NurecRestartRequest,
    NurecRestartResponse,
    LoadRequest,
    LoadResponse,
    PoseAtTime,
    RenderRequest,
    RenderResponse,
    ScenarioInfo,
    TimeRange,
    TrajectoryPoint,
    TrajectoryResponse,
    UsdzFilesResponse,
)
from scenario_runtime import PlaybackClock, Scenario

# Load protobuf modules generated from local ./proto at runtime.
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(BACKEND_DIR)
PROTO_SRC_DIR = os.path.join(REPO_DIR, "proto")
PROTO_GEN_DIR = os.path.join(BACKEND_DIR, ".cache", "proto_generated")
DEFAULT_SAMPLE_SET_DIR = os.path.join(REPO_DIR, "sample_set")


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
        self.runtime_scenario: Optional[Scenario] = None
        self.playback: Optional[PlaybackClock] = None


app_state = AppState()


def _resolve_usdz_path(raw_path: str) -> str:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = Path(REPO_DIR) / path
    return str(path.resolve())


def _list_usdz_files(base_dir: str) -> List[str]:
    root = Path(base_dir)
    if not root.exists():
        return []
    files = sorted(str(p.resolve()) for p in root.rglob("*.usdz") if p.is_file())
    return files


def _detect_docker_command() -> List[str]:
    normal = subprocess.run(
        ["docker", "ps"],
        check=False,
        capture_output=True,
        text=True,
    )
    if normal.returncode == 0:
        return ["docker"]
    with_sudo = subprocess.run(
        ["sudo", "-n", "docker", "ps"],
        check=False,
        capture_output=True,
        text=True,
    )
    if with_sudo.returncode == 0:
        return ["sudo", "-n", "docker"]
    raise RuntimeError(
        "Docker is not accessible. Configure docker group access or passwordless sudo."
    )


def _restart_nurec_container(usdz_path: str, port: int) -> str:
    image = os.getenv("NUREC_IMAGE", "carlasimulator/nvidia-nurec-grpc:0.2.0")
    gpu_id = os.getenv("CUDA_VISIBLE_DEVICES", "0")
    pytorch_alloc_conf = os.getenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    container_name = os.getenv("NUREC_CONTAINER_NAME", f"nurec-grpc-{port}")
    docker = _detect_docker_command()

    subprocess.run(
        docker + ["rm", "-f", container_name],
        check=False,
        capture_output=True,
        text=True,
    )

    usdz_dir = str(Path(usdz_path).resolve().parent)
    run_cmd = docker + [
        "run",
        "-d",
        "--rm",
        "--name",
        container_name,
        "--gpus",
        "all",
        "--ipc=host",
        "-e",
        f"CUDA_VISIBLE_DEVICES={gpu_id}",
        "-e",
        f"PYTORCH_CUDA_ALLOC_CONF={pytorch_alloc_conf}",
        "-p",
        f"{port}:{port}",
        "-v",
        f"{usdz_dir}:{usdz_dir}:ro",
        image,
        "--artifact-glob",
        usdz_path,
        "--port",
        str(port),
        "--host",
        "0.0.0.0",
        "--test-scenes-are-valid",
    ]
    proc = subprocess.run(run_cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "Failed to start NuRec docker container")
    return container_name


def _probe_nurec_connection(host: str, port: int, timeout_seconds: float = 0.8) -> bool:
    """Check if NuRec gRPC endpoint is reachable."""
    try:
        channel = grpc.insecure_channel(f"{host}:{port}")
        grpc.channel_ready_future(channel).result(timeout=timeout_seconds)
        channel.close()
        return True
    except Exception:
        return False


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


def _matrix44_to_pose(matrix_data: List[List[float]]):
    matrix = np.array(matrix_data, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected 4x4 transform matrix, got shape {matrix.shape}")
    return _matrix_to_pose(matrix)


def _parse_shutter_type(shutter_type_raw: Any):
    if not isinstance(shutter_type_raw, str):
        return sensorsim_pb2.ShutterType.UNKNOWN
    normalized = shutter_type_raw.strip().upper()
    return sensorsim_pb2.ShutterType.Value(normalized) if normalized in sensorsim_pb2.ShutterType.keys() else sensorsim_pb2.ShutterType.UNKNOWN


def _parse_reference_poly(reference_poly_raw: Any):
    if not isinstance(reference_poly_raw, str):
        return sensorsim_pb2.FthetaCameraParam.PolynomialType.UNKNOWN
    normalized = reference_poly_raw.strip().upper()
    return (
        sensorsim_pb2.FthetaCameraParam.PolynomialType.Value(normalized)
        if normalized in sensorsim_pb2.FthetaCameraParam.PolynomialType.keys()
        else sensorsim_pb2.FthetaCameraParam.PolynomialType.UNKNOWN
    )


def _build_camera_spec(logical_id: str, calibration: Dict[str, Any]):
    model = calibration.get("camera_model", {})
    params = model.get("parameters", {})
    resolution = params.get("resolution", [1080, 1920])
    if not isinstance(resolution, list) or len(resolution) != 2:
        raise ValueError(f"Invalid camera resolution for {logical_id}: {resolution}")

    camera_spec = sensorsim_pb2.CameraSpec(
        logical_id=logical_id,
        trajectory_idx=int(calibration.get("trajectory_idx", 0)),
        resolution_h=int(resolution[0]),
        resolution_w=int(resolution[1]),
        shutter_type=_parse_shutter_type(params.get("shutter_type")),
    )

    principal_point = params.get("principal_point", [0.0, 0.0])
    linear_cde = params.get("linear_cde", [0.0, 0.0, 0.0])
    if isinstance(linear_cde, dict):
        linear_cde = [
            float(linear_cde.get("linear_c", 0.0)),
            float(linear_cde.get("linear_d", 0.0)),
            float(linear_cde.get("linear_e", 0.0)),
        ]
    while len(linear_cde) < 3:
        linear_cde.append(0.0)

    camera_spec.ftheta_param.principal_point_x = float(principal_point[0]) if len(principal_point) > 0 else 0.0
    camera_spec.ftheta_param.principal_point_y = float(principal_point[1]) if len(principal_point) > 1 else 0.0
    camera_spec.ftheta_param.reference_poly = _parse_reference_poly(params.get("reference_poly"))
    camera_spec.ftheta_param.pixeldist_to_angle_poly.extend(params.get("pixeldist_to_angle_poly", []))
    camera_spec.ftheta_param.angle_to_pixeldist_poly.extend(params.get("angle_to_pixeldist_poly", []))
    camera_spec.ftheta_param.max_angle = float(params.get("max_angle", 0.0))
    camera_spec.ftheta_param.linear_cde.linear_c = float(linear_cde[0])
    camera_spec.ftheta_param.linear_cde.linear_d = float(linear_cde[1])
    camera_spec.ftheta_param.linear_cde.linear_e = float(linear_cde[2])
    return camera_spec


def _load_scenario_from_usdz(usdz_path: str) -> tuple[str, str, Dict[str, object], List[object]]:
    from scenario_runtime import extract_json_from_usdz

    json_array = extract_json_from_usdz(
        usdz_path,
        ["rig_trajectories.json", "sequence_tracks.json", "data_info.json"],
    )
    missing = {"rig_trajectories.json", "data_info.json"} - set(json_array.keys())
    if missing:
        raise ValueError(f"USDZ missing required files: {sorted(missing)}")

    rig_data = json_array["rig_trajectories.json"]
    data_info = json_array["data_info.json"]
    if not isinstance(rig_data, dict) or not isinstance(data_info, dict):
        raise ValueError("Invalid JSON structure in USDZ.")

    file_stem = os.path.splitext(os.path.basename(usdz_path))[0]
    sequence_id = str(data_info.get("sequence_id", file_stem))
    scene_id = sequence_id

    camera_calibrations = rig_data.get("camera_calibrations", {})
    cameras: Dict[str, object] = {}
    for camera_key, calibration in camera_calibrations.items():
        logical_id = str(calibration.get("logical_sensor_name", camera_key))
        camera_spec = _build_camera_spec(logical_id, calibration)
        rig_to_camera = _matrix44_to_pose(calibration["T_sensor_rig"])
        available_camera = sensorsim_pb2.AvailableCamerasReturn.AvailableCamera(
            intrinsics=camera_spec,
            rig_to_camera=rig_to_camera,
            logical_id=logical_id,
            trajectory_idx=int(calibration.get("trajectory_idx", 0)),
        )
        cameras[logical_id] = available_camera

    rig_trajectories = rig_data.get("rig_trajectories", [])
    trajectories: List[object] = []
    for trajectory_idx, rig_trajectory in enumerate(rig_trajectories):
        timestamps = rig_trajectory.get("T_rig_world_timestamps_us", [])
        transforms = rig_trajectory.get("T_rig_worlds", [])
        if len(timestamps) != len(transforms):
            logger.warning(
                "Trajectory %s has mismatched timestamp/pose counts (%s/%s), truncating.",
                trajectory_idx,
                len(timestamps),
                len(transforms),
            )
        pose_count = min(len(timestamps), len(transforms))
        trajectory_msg = common_pb2.Trajectory()
        for idx in range(pose_count):
            trajectory_msg.poses.add(
                timestamp_us=int(timestamps[idx]),
                pose=_matrix44_to_pose(transforms[idx]),
            )
        trajectories.append(
            sensorsim_pb2.AvailableTrajectoriesReturn.AvailableTrajectory(
                trajectory_idx=trajectory_idx,
                trajectory=trajectory_msg,
            )
        )

    if not cameras:
        raise ValueError("No camera calibrations found in USDZ.")
    if not trajectories:
        raise ValueError("No rig trajectories found in USDZ.")
    return scene_id, sequence_id, cameras, trajectories


def _get_runtime_scenario() -> Scenario:
    if app_state.runtime_scenario is None:
        raise HTTPException(status_code=400, detail="No scenario loaded. Call /api/load first.")
    return app_state.runtime_scenario


def _get_playback() -> PlaybackClock:
    if app_state.playback is None:
        raise HTTPException(status_code=400, detail="Playback not initialized. Call /api/load first.")
    return app_state.playback


def _playback_state_response(playback: PlaybackClock) -> PlaybackStateResponse:
    scenario = playback.scenario
    pose_range = scenario.metadata.get("pose-range", {})
    end_ts = int(pose_range.get("end-timestamp_us", scenario.tracks.current_time))
    return PlaybackStateResponse(
        is_playing=playback.is_playing,
        speed=float(playback.speed),
        current_time_us=int(scenario.tracks.current_time),
        seconds_since_start=float(scenario.tracks.get_current_time_seconds()),
        done=int(scenario.tracks.current_time) >= end_ts,
    )


@app.get("/api/health")
async def health_check():
    grpc_connected = _probe_nurec_connection(app_state.nurec_host, app_state.nurec_port)
    return {
        "status": "healthy",
        "nurec_available": NUREC_AVAILABLE,
        "scenario_loaded": app_state.scene_id is not None,
        "grpc_connected": grpc_connected,
    }


@app.get("/api/usdz-files", response_model=UsdzFilesResponse)
async def get_usdz_files():
    files = _list_usdz_files(DEFAULT_SAMPLE_SET_DIR)
    return UsdzFilesResponse(files=files)


@app.post("/api/nurec/restart", response_model=NurecRestartResponse)
async def restart_nurec(request: NurecRestartRequest):
    usdz_path = _resolve_usdz_path(request.usdz_path)
    if not os.path.exists(usdz_path):
        raise HTTPException(status_code=404, detail=f"USDZ file not found: {usdz_path}")

    try:
        container_name = _restart_nurec_container(usdz_path, request.nurec_port)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to restart NuRec: {exc}")

    app_state.nurec_host = request.nurec_host
    app_state.nurec_port = request.nurec_port
    if app_state.grpc_channel:
        app_state.grpc_channel.close()
    app_state.grpc_channel = None
    app_state.grpc_stub = None
    app_state.scene_id = None
    app_state.available_cameras = {}
    app_state.available_trajectories = []
    app_state.runtime_scenario = None
    app_state.playback = None

    grpc_ready = False
    for _ in range(45):
        if _probe_nurec_connection(request.nurec_host, request.nurec_port):
            grpc_ready = True
            break
        time.sleep(1)

    return NurecRestartResponse(
        status="restarted" if grpc_ready else "starting",
        container_name=container_name,
        usdz_path=usdz_path,
        grpc_ready=grpc_ready,
    )


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
        if app_state.grpc_channel:
            app_state.grpc_channel.close()
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
        grpc.channel_ready_future(app_state.grpc_channel).result(timeout=5.0)

        scene_id, sequence_id, cameras, trajectories = _load_scenario_from_usdz(request.usdz_path)
        app_state.scene_id = scene_id
        app_state.available_cameras = cameras
        app_state.available_trajectories = trajectories
        app_state.runtime_scenario = Scenario(request.usdz_path)
        app_state.playback = PlaybackClock(app_state.runtime_scenario)

        if sequence_id != scene_id:
            logger.info(
                "USDZ sequence_id (%s) differs from scene_id derived from filename (%s).",
                sequence_id,
                scene_id,
            )
        return LoadResponse(status="loaded", sequence_id=sequence_id)
    except Exception as e:
        logger.error(f"Failed to load scenario: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/scenario", response_model=ScenarioInfo)
async def get_scenario_info():
    scenario = _get_runtime_scenario()
    pose_range = scenario.metadata.get("pose-range", {})
    start_us = int(pose_range.get("start-timestamp_us", 0))
    end_us = int(pose_range.get("end-timestamp_us", start_us))
    time_range = TimeRange(
        start_us=start_us,
        end_us=end_us,
        duration_seconds=(end_us - start_us) / 1_000_000 if end_us > start_us else 0.0,
    )

    cameras = []
    for calibration in scenario.camera_calibrations.values():
        logical_id = calibration.logical_sensor_name
        resolution = calibration.camera_model.parameters.resolution
        cameras.append(
            CameraInfo(
                logical_name=logical_id,
                resolution=[int(resolution[1]), int(resolution[0])],
                T_sensor_rig=calibration.T_sensor_rig,
            )
        )

    return ScenarioInfo(sequence_id=app_state.scene_id or "", time_range=time_range, cameras=cameras)


@app.get("/api/trajectory", response_model=TrajectoryResponse)
async def get_trajectory(sample_interval_us: int = Query(default=100000, ge=10000)):
    scenario = _get_runtime_scenario()
    pose_range = scenario.metadata.get("pose-range", {})
    start_us = int(pose_range.get("start-timestamp_us", int(scenario.ego_poses.start_time())))
    end_us = int(pose_range.get("end-timestamp_us", int(scenario.ego_poses.end_time())))
    points = []
    for ts in range(start_us, end_us + 1, sample_interval_us):
        pose = scenario.ego_poses.interpolate_pose_xyzquat(ts)
        if pose is None:
            continue
        points.append(
            TrajectoryPoint(
                timestamp_us=ts,
                position=[float(pose[0]), float(pose[1]), float(pose[2])],
                quaternion=[float(pose[3]), float(pose[4]), float(pose[5]), float(pose[6])],
            )
        )
    return TrajectoryResponse(trajectory=points)


@app.get("/api/pose", response_model=PoseAtTime)
async def get_pose_at_time(timestamp_us: int = Query(...)):
    scenario = _get_runtime_scenario()
    pose = scenario.ego_poses.interpolate_pose_xyzquat(timestamp_us)
    if pose is None:
        raise HTTPException(status_code=400, detail="Timestamp out of range")
    return PoseAtTime(
        timestamp_us=timestamp_us,
        position=[float(pose[0]), float(pose[1]), float(pose[2])],
        quaternion=[float(pose[3]), float(pose[4]), float(pose[5]), float(pose[6])],
    )


@app.post("/api/render", response_model=RenderResponse)
async def render_cameras(request: RenderRequest):
    if app_state.grpc_stub is None or app_state.scene_id is None:
        raise HTTPException(status_code=400, detail="No scenario loaded or NuRec not connected")

    scenario = _get_runtime_scenario()
    rig_matrix = scenario.ego_poses.interpolate_pose_matrix(request.timestamp_us)
    if rig_matrix is None:
        raise HTTPException(status_code=400, detail="Timestamp out of range")
    results = {}

    for camera_id in request.camera_ids:
        cam = app_state.available_cameras.get(camera_id)
        if cam is None:
            logger.warning(f"Camera {camera_id} not found in available cameras")
            continue

        camera_matrix = rig_matrix @ _pose_to_matrix(cam.rig_to_camera)
        sensor_pose = _matrix_to_pose(camera_matrix)

        # Keep calibration intrinsics unchanged; only scale output resolution.
        # For fisheye/ftheta cameras, mutating intrinsic resolution can behave like FOV crop.
        camera_spec = sensorsim_pb2.CameraSpec()
        camera_spec.CopyFrom(cam.intrinsics)
        output_h = max(1, int(camera_spec.resolution_h * request.resolution_scale))
        output_w = max(1, int(camera_spec.resolution_w * request.resolution_scale))

        grpc_request = sensorsim_pb2.RGBRenderRequest(
            scene_id=app_state.scene_id,
            resolution_h=output_h,
            resolution_w=output_w,
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


@app.get("/api/playback", response_model=PlaybackStateResponse)
async def get_playback_state():
    playback = _get_playback()
    return _playback_state_response(playback)


@app.post("/api/playback/play", response_model=PlaybackStateResponse)
async def playback_play(request: PlaybackControlRequest):
    playback = _get_playback()
    if request.speed is not None:
        if request.speed <= 0:
            raise HTTPException(status_code=400, detail="speed must be > 0")
        playback.speed = float(request.speed)
    playback.play()
    return _playback_state_response(playback)


@app.post("/api/playback/stop", response_model=PlaybackStateResponse)
async def playback_stop():
    playback = _get_playback()
    playback.stop()
    return _playback_state_response(playback)


@app.post("/api/playback/reset", response_model=PlaybackStateResponse)
async def playback_reset():
    playback = _get_playback()
    playback.reset()
    return _playback_state_response(playback)


@app.post("/api/playback/tick", response_model=PlaybackTickResponse)
async def playback_tick(request: PlaybackTickRequest):
    playback = _get_playback()
    new_track_ids, removed_track_ids, current_time_us = playback.tick(request.delta_us)
    state = _playback_state_response(playback)
    return PlaybackTickResponse(
        is_playing=state.is_playing,
        speed=state.speed,
        current_time_us=current_time_us,
        seconds_since_start=state.seconds_since_start,
        done=state.done,
        new_track_ids=new_track_ids,
        removed_track_ids=removed_track_ids,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
