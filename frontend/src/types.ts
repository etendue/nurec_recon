// SPDX-FileCopyrightText: © 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * TypeScript type definitions for NuRec Web Viewer
 */

export interface TimeRange {
  start_us: number;
  end_us: number;
  duration_seconds: number;
}

export interface CameraInfo {
  logical_name: string;
  resolution: [number, number];
  T_sensor_rig: number[][];
}

export interface ScenarioInfo {
  sequence_id: string;
  time_range: TimeRange;
  cameras: CameraInfo[];
}

export interface LoadRequest {
  usdz_path: string;
  nurec_host?: string;
  nurec_port?: number;
}

export interface LoadResponse {
  status: string;
  sequence_id: string;
}

export interface PoseAtTime {
  timestamp_us: number;
  position: [number, number, number];
  quaternion: [number, number, number, number];
}

export interface TrajectoryPoint {
  timestamp_us: number;
  position: [number, number, number];
  quaternion: [number, number, number, number];
}

export interface TrajectoryResponse {
  trajectory: TrajectoryPoint[];
}

export interface RenderRequest {
  timestamp_us: number;
  camera_ids: string[];
  resolution_scale?: number;
}

export interface RenderResponse {
  timestamp_us: number;
  images: Record<string, string>; // camera_id -> base64 JPEG
}

export interface PlaybackState {
  isPlaying: boolean;
  currentTime_us: number;
  playbackSpeed: number; // 0.5, 1, 2
}

export interface HealthResponse {
  status: string;
  nurec_available: boolean;
  scenario_loaded: boolean;
  grpc_connected: boolean;
}
