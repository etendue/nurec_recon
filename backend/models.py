# SPDX-FileCopyrightText: © 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Pydantic models for NuRec Web Viewer API
"""

from pydantic import BaseModel
from typing import List, Dict, Optional


class TimeRange(BaseModel):
    """Time range information for a scenario"""
    start_us: int
    end_us: int
    duration_seconds: float


class CameraInfo(BaseModel):
    """Camera information from scenario"""
    logical_name: str
    resolution: List[int]
    T_sensor_rig: List[List[float]]


class ScenarioInfo(BaseModel):
    """Complete scenario information"""
    sequence_id: str
    time_range: TimeRange
    cameras: List[CameraInfo]


class LoadRequest(BaseModel):
    """Request to load a scenario"""
    usdz_path: str
    nurec_host: str = "localhost"
    nurec_port: int = 46435


class LoadResponse(BaseModel):
    """Response after loading a scenario"""
    status: str
    sequence_id: str


class PoseAtTime(BaseModel):
    """Pose at a specific timestamp"""
    timestamp_us: int
    position: List[float]  # [x, y, z]
    quaternion: List[float]  # [qx, qy, qz, qw]


class TrajectoryPoint(BaseModel):
    """Single point in a trajectory"""
    timestamp_us: int
    position: List[float]
    quaternion: List[float]


class TrajectoryResponse(BaseModel):
    """Complete trajectory response"""
    trajectory: List[TrajectoryPoint]


class RenderRequest(BaseModel):
    """Request to render camera images"""
    timestamp_us: int
    camera_ids: List[str]
    resolution_scale: float = 0.25


class RenderResponse(BaseModel):
    """Response with rendered images"""
    timestamp_us: int
    images: Dict[str, str]  # camera_id -> base64 encoded JPEG


class ErrorResponse(BaseModel):
    """Error response"""
    detail: str


class PlaybackControlRequest(BaseModel):
    """Playback control request"""
    speed: Optional[float] = None


class PlaybackTickRequest(BaseModel):
    """Manual tick request"""
    delta_us: Optional[int] = None


class PlaybackStateResponse(BaseModel):
    """Current playback state"""
    is_playing: bool
    speed: float
    current_time_us: int
    seconds_since_start: float
    done: bool


class PlaybackTickResponse(PlaybackStateResponse):
    """Tick update response"""
    new_track_ids: List[str]
    removed_track_ids: List[str]


class UsdzFilesResponse(BaseModel):
    """USDZ file list from default sample directory"""
    files: List[str]


class NurecRestartRequest(BaseModel):
    """Request to restart NuRec docker service with a specific USDZ"""
    usdz_path: str
    nurec_host: str = "localhost"
    nurec_port: int = 46435


class NurecRestartResponse(BaseModel):
    """Response after NuRec restart request"""
    status: str
    container_name: str
    usdz_path: str
    grpc_ready: bool
