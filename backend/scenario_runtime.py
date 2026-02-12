from __future__ import annotations

import json
import logging
import time
import zipfile
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

EGO_TRACK_ID = "ego"
EGO_LABEL = "vehicle.mercedes.coupe_2020"
EGO_DIMS = None
EGO_FLAG = "EGO"

DYNAMIC_FLAG = "DYNAMIC"
CONTROLLABLE_FLAG = "CONTROLLABLE"

SPECTATOR_TRACK_ID = "spectator"
SPECTATOR_LABEL = "spectator"
SPECTATOR_FLAG = "SPECTATOR"


def extract_json_from_usdz(usdz_file: str, json_files: List[str]) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    with zipfile.ZipFile(usdz_file, "r") as zip_ref:
        for file_info in zip_ref.infolist():
            if file_info.filename in json_files:
                with zip_ref.open(file_info.filename) as source_file:
                    results[file_info.filename] = json.load(source_file)
    return results


def lowpass_filter_vertical_component(
    poses: List[np.ndarray],
    threshold: float = 0.01,
    max_slope: float = 0.25,
    window_size: int = 5,
    cutoff_freq: float = 0.05,
) -> List[np.ndarray]:
    if len(poses) < 3:
        return poses

    poses_array = np.array(poses)
    n_poses = len(poses_array)
    positions = poses_array[:, :3, 3]

    horizontal_distances_list = []
    grades_list = []
    for i in range(n_poses - 1):
        dx = positions[i + 1, 0] - positions[i, 0]
        dy = positions[i + 1, 1] - positions[i, 1]
        dz = positions[i + 1, 2] - positions[i, 2]
        horizontal_distance = np.sqrt(dx**2 + dy**2)
        horizontal_distances_list.append(horizontal_distance)
        grade = dz / horizontal_distance if horizontal_distance > threshold else 0.0
        grades_list.append(grade)

    horizontal_distances = np.array(horizontal_distances_list)
    grades = np.array(grades_list)
    constrained_grades = np.copy(grades)
    for i in range(len(grades)):
        if abs(constrained_grades[i]) > max_slope:
            constrained_grades[i] = np.sign(constrained_grades[i]) * max_slope

    filtered_grades = constrained_grades.copy()
    if len(constrained_grades) > 10:
        cumulative_distances = np.cumsum(np.concatenate([[0], horizontal_distances]))
        if window_size > 1:
            kernel = np.ones(window_size) / window_size
            filtered_grades = np.convolve(constrained_grades, kernel, mode="same")

        if len(filtered_grades) > 20:
            try:
                distances_for_interp = cumulative_distances[1:]
                unique_distances, unique_indices = np.unique(
                    distances_for_interp, return_index=True
                )
                if len(unique_distances) > 3:
                    unique_grades = filtered_grades[unique_indices]
                    uniform_distances = np.linspace(
                        unique_distances[0], unique_distances[-1], len(unique_distances)
                    )
                    interp_func = interp1d(
                        unique_distances,
                        unique_grades,
                        kind="linear",
                        bounds_error=False,
                        fill_value="extrapolate",
                    )
                    uniform_grades = interp_func(uniform_distances)
                    b, a = butter(3, cutoff_freq, btype="low")
                    uniform_grades_filtered = filtfilt(b, a, uniform_grades)
                    interp_back_func = interp1d(
                        uniform_distances,
                        uniform_grades_filtered,
                        kind="linear",
                        bounds_error=False,
                        fill_value="extrapolate",
                    )
                    filtered_grades = interp_back_func(distances_for_interp)
            except Exception as exc:
                logger.warning("Could not apply grade lowpass filter: %s", exc)

    filtered_positions = positions.copy()
    filtered_positions[0, 2] = positions[0, 2]
    for i in range(1, n_poses):
        horizontal_distance = horizontal_distances[i - 1]
        filtered_positions[i, 2] = (
            filtered_positions[i - 1, 2] + filtered_grades[i - 1] * horizontal_distance
        )

    filtered_poses = []
    for i in range(n_poses):
        pose = poses_array[i].copy()
        pose[:3, 3] = filtered_positions[i]
        filtered_poses.append(pose)
    return filtered_poses


class PoseType(Enum):
    TRANSFORM_MATRIX = "transform_matrix"
    XYZ_QUAT = "xyz_quat"


class InterpolatedPoses:
    def __init__(
        self,
        poses: List[Union[np.ndarray, List[float]]],
        timestamps: List[float],
        pose_type: PoseType = PoseType.TRANSFORM_MATRIX,
        filter_vertical_poses: bool = False,
    ) -> None:
        self.poses: List[np.ndarray] = []
        self._convert_poses_to_mats(poses, pose_type, filter_vertical_poses)
        self.timestamps = timestamps
        self.ignore_out_of_bounds = False
        self.transform = np.eye(4)

    def _convert_poses_to_mats(
        self,
        poses: List[Union[np.ndarray, List[float]]],
        pose_type: PoseType,
        filter_vertical_poses: bool = False,
    ) -> None:
        for pose in poses:
            if pose_type == PoseType.TRANSFORM_MATRIX:
                matrix = np.array(pose)
                if matrix.shape != (4, 4):
                    raise ValueError(f"Expected 4x4 matrix but got shape {matrix.shape}")
                self.poses.append(matrix)
            elif pose_type == PoseType.XYZ_QUAT:
                pose_array = np.array(pose)
                translation = pose_array[:3]
                quaternion = pose_array[3:7]
                rot_matrix = Rotation.from_quat(quaternion).as_matrix()
                transform = np.eye(4)
                transform[:3, :3] = rot_matrix
                transform[:3, 3] = translation
                self.poses.append(transform)
            else:
                raise ValueError(f"Unsupported pose type: {pose_type}")

        if filter_vertical_poses:
            self.poses = lowpass_filter_vertical_component(self.poses)

    def set_ignore_out_of_bounds(self, ignore_out_of_bounds: bool) -> None:
        self.ignore_out_of_bounds = ignore_out_of_bounds

    def set_transform(self, transform: np.ndarray) -> None:
        self.transform = transform

    def _get_interpolation_params(
        self, timestamp: float
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        if self.ignore_out_of_bounds and timestamp > self.timestamps[-1]:
            return self.poses[-2], self.poses[-1], 1.0
        if timestamp < self.timestamps[0] or timestamp > self.timestamps[-1]:
            return None, None, 0.0

        prev_timestamp_idx = 0
        for i in range(len(self.timestamps)):
            if self.timestamps[i] > timestamp:
                break
            prev_timestamp_idx = i
        next_timestamp_idx = prev_timestamp_idx + 1

        if next_timestamp_idx >= len(self.timestamps):
            return self.poses[prev_timestamp_idx], None, 0.0

        t = (timestamp - self.timestamps[prev_timestamp_idx]) / (
            self.timestamps[next_timestamp_idx] - self.timestamps[prev_timestamp_idx]
        )
        start_pose = self.poses[prev_timestamp_idx]
        end_pose = self.poses[next_timestamp_idx]
        return start_pose, end_pose, t

    def interpolate_pose_matrix(self, timestamp: float) -> Optional[np.ndarray]:
        start_pose, end_pose, t = self._get_interpolation_params(timestamp)
        if start_pose is None:
            return None
        if end_pose is None:
            return self.transform @ start_pose

        start_rotation = Rotation.from_matrix(start_pose[:3, :3])
        end_rotation = Rotation.from_matrix(end_pose[:3, :3])
        rotvec = (start_rotation.inv() * end_rotation).as_rotvec()
        slerp_result = start_rotation * Rotation.from_rotvec(rotvec * t)

        interp_translation = start_pose[:3, 3] + t * (end_pose[:3, 3] - start_pose[:3, 3])
        result = np.eye(4)
        result[:3, :3] = slerp_result.as_matrix()
        result[:3, 3] = interp_translation
        return self.transform @ result

    def interpolate_pose_xyzquat(self, timestamp: float) -> Optional[List[float]]:
        matrix = self.interpolate_pose_matrix(timestamp)
        if matrix is None:
            return None
        translation = matrix[:3, 3]
        quaternion = Rotation.from_matrix(matrix[:3, :3]).as_quat()
        return np.concatenate([translation, quaternion]).tolist()


class Track(InterpolatedPoses):
    def __init__(
        self,
        track_id: str,
        poses: List[Union[np.ndarray, List[float]]],
        timestamps: List[float],
        dims: Optional[List[float]],
        label: str,
        flags: List[str],
        pose_type: PoseType = PoseType.XYZ_QUAT,
        filter_vertical_poses: bool = False,
    ) -> None:
        super().__init__(poses, timestamps, pose_type, filter_vertical_poses)
        self.track_id = track_id
        self.dims = dims
        self.label = label
        self.dynamic = DYNAMIC_FLAG in flags
        self.controllable = CONTROLLABLE_FLAG in flags
        self.ego = EGO_FLAG in flags
        self.spectator = SPECTATOR_FLAG in flags

    def start_time(self) -> float:
        return self.timestamps[0]

    def end_time(self) -> float:
        return self.timestamps[-1]


def extract_poses_from_json(
    json_array: Dict[str, Any], filter_vertical_poses: bool = False
) -> List[Track]:
    track_data: List[Track] = []
    tracks_data = json_array["sequence_tracks.json"]["dummy_chunk_id"]["tracks_data"]
    cuboids_data = json_array["sequence_tracks.json"]["dummy_chunk_id"]["cuboidtracks_data"]
    for i, track_id in enumerate(tracks_data["tracks_id"]):
        track_data.append(
            Track(
                track_id,
                tracks_data["tracks_poses"][i],
                tracks_data["tracks_timestamps_us"][i],
                cuboids_data["cuboids_dims"][i],
                tracks_data["tracks_label_class"][i],
                tracks_data["tracks_flags"][i],
                filter_vertical_poses=filter_vertical_poses,
            )
        )
    track_data.sort(key=lambda x: x.start_time())
    return track_data


class Tracks:
    def __init__(self, track_data: List[Track], zero_time: int) -> None:
        track_data.sort(key=lambda x: x.start_time())
        self.track_data = track_data
        self.zero_time = zero_time
        self.current_time = zero_time
        self.active_tracks: List[Track] = []
        self.track_index = 0
        self.min_lifetime = int(0.1 * 1e6)

    def reset(self) -> None:
        self.current_time = self.zero_time
        self.active_tracks = []
        self.track_index = 0

    def update(self, time_step: float) -> Tuple[List[Track], List[Track]]:
        self.current_time += time_step
        tracks_to_remove = [t for t in self.active_tracks if self.current_time > t.end_time()]
        for track in tracks_to_remove:
            self.active_tracks.remove(track)

        new_tracks: List[Track] = []
        while (
            self.track_index < len(self.track_data)
            and self.track_data[self.track_index].start_time() <= self.current_time
        ):
            next_track = self.track_data[self.track_index]
            lifetime = next_track.end_time() - next_track.start_time()
            if lifetime > self.min_lifetime:
                self.active_tracks.append(next_track)
                new_tracks.append(next_track)
            self.track_index += 1
        return new_tracks, tracks_to_remove

    def get_current_time_seconds(self) -> float:
        return (self.current_time - self.zero_time) / 1e6

    def set_minimum_lifetime(self, min_lifetime_seconds: float) -> None:
        self.min_lifetime = int(min_lifetime_seconds * 1e6)

    def get_all_possible_tracks(self) -> List[Track]:
        return [
            track
            for track in self.track_data
            if track.end_time() - track.start_time() > self.min_lifetime
        ]

    def set_view_transform(self, transform: np.ndarray) -> None:
        for track in self.track_data:
            track.set_transform(transform)


def get_best_camera(rig_trajectories: Dict[str, Any]) -> Dict[str, Any]:
    cameras = rig_trajectories["camera_calibrations"]
    camera_names = [
        (camera_data["logical_sensor_name"], camera_data)
        for _, camera_data in cameras.items()
    ]
    best_camera = None
    search_patterns = ["front_wide_120fov", "front_wide", "front", "wide", "120"]
    for pattern in search_patterns:
        for camera_name, camera_data in camera_names:
            if pattern in camera_name:
                best_camera = camera_data
                break
        if best_camera is not None:
            break
    if best_camera is None:
        best_camera = camera_names[0][1]
    return best_camera


def get_spectator(rig_trajectories: Dict[str, Any], ego_poses: Track) -> Track:
    best_camera = get_best_camera(rig_trajectories)
    transformed_poses = np.array(ego_poses.poses) @ np.array(best_camera["T_sensor_rig"])
    return Track(
        SPECTATOR_TRACK_ID,
        transformed_poses,
        ego_poses.timestamps,
        None,
        SPECTATOR_LABEL,
        [SPECTATOR_FLAG],
        PoseType.TRANSFORM_MATRIX,
    )


@dataclass
class CameraModelParameters:
    resolution: List[int]
    shutter_type: str
    external_distortion_parameters: Optional[Any]
    principal_point: List[float]
    reference_poly: str
    pixeldist_to_angle_poly: List[float]
    angle_to_pixeldist_poly: List[float]
    max_angle: float
    linear_cde: List[float]


@dataclass
class CameraModel:
    type: str
    parameters: CameraModelParameters


@dataclass
class CameraCalibration:
    sequence_id: str
    logical_sensor_name: str
    unique_sensor_idx: int
    T_sensor_rig: List[List[float]]
    camera_model: CameraModel


def get_camera_calibrations(json_array: Dict[str, Any]) -> Dict[str, CameraCalibration]:
    camera_calibrations = json_array["rig_trajectories.json"]["camera_calibrations"]
    result: Dict[str, CameraCalibration] = {}
    for camera_id, calibration in camera_calibrations.items():
        model_params = calibration["camera_model"]["parameters"]
        camera_model = CameraModel(
            type=calibration["camera_model"]["type"],
            parameters=CameraModelParameters(
                resolution=model_params["resolution"],
                shutter_type=model_params["shutter_type"],
                external_distortion_parameters=model_params.get("external_distortion_parameters"),
                principal_point=model_params["principal_point"],
                reference_poly=model_params["reference_poly"],
                pixeldist_to_angle_poly=model_params["pixeldist_to_angle_poly"],
                angle_to_pixeldist_poly=model_params["angle_to_pixeldist_poly"],
                max_angle=model_params["max_angle"],
                linear_cde=model_params["linear_cde"],
            ),
        )
        result[camera_id] = CameraCalibration(
            sequence_id=calibration["sequence_id"],
            logical_sensor_name=calibration["logical_sensor_name"],
            unique_sensor_idx=calibration["unique_sensor_idx"],
            T_sensor_rig=calibration["T_sensor_rig"],
            camera_model=camera_model,
        )
    return result


class Scenario:
    def __init__(self, usdz_file: str) -> None:
        json_array = extract_json_from_usdz(
            usdz_file,
            ["rig_trajectories.json", "sequence_tracks.json", "data_info.json"],
        )
        self.metadata = json_array["data_info.json"]
        track_data = extract_poses_from_json(json_array, filter_vertical_poses=True)
        self.camera_calibrations: Dict[str, CameraCalibration] = get_camera_calibrations(
            json_array
        )
        self.t_world_base = np.array(json_array["rig_trajectories.json"]["T_world_base"])
        self.ego_poses = Track(
            EGO_TRACK_ID,
            json_array["rig_trajectories.json"]["rig_trajectories"][0]["T_rig_worlds"],
            json_array["rig_trajectories.json"]["rig_trajectories"][0][
                "T_rig_world_timestamps_us"
            ],
            EGO_DIMS,
            EGO_LABEL,
            [EGO_FLAG, DYNAMIC_FLAG],
            PoseType.TRANSFORM_MATRIX,
        )
        self.spectator = get_spectator(json_array["rig_trajectories.json"], self.ego_poses)
        self.controllable_tracks = {
            track.track_id for track in track_data if track.controllable
        }
        self.tracks = Tracks(
            track_data, self.metadata["pose-range"]["start-timestamp_us"]
        )


class PlaybackClock:
    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario
        self.is_playing = False
        self.speed = 1.0
        self.last_wall_time = time.monotonic()

    def reset(self) -> None:
        self.scenario.tracks.reset()
        self.is_playing = False
        self.last_wall_time = time.monotonic()

    def play(self) -> None:
        self.is_playing = True
        self.last_wall_time = time.monotonic()

    def stop(self) -> None:
        self.is_playing = False

    def tick(self, delta_us: Optional[int] = None) -> Tuple[List[str], List[str], int]:
        if not self.is_playing and delta_us is None:
            return [], [], int(self.scenario.tracks.current_time)

        if delta_us is None:
            now = time.monotonic()
            delta_us = int((now - self.last_wall_time) * 1_000_000 * self.speed)
            self.last_wall_time = now
        else:
            delta_us = int(delta_us * self.speed)

        if delta_us <= 0:
            return [], [], int(self.scenario.tracks.current_time)

        new_tracks, removed_tracks = self.scenario.tracks.update(delta_us)
        end_ts = int(self.scenario.metadata["pose-range"]["end-timestamp_us"])
        if self.scenario.tracks.current_time >= end_ts:
            self.scenario.tracks.current_time = end_ts
            self.is_playing = False

        return (
            [track.track_id for track in new_tracks],
            [track.track_id for track in removed_tracks],
            int(self.scenario.tracks.current_time),
        )
