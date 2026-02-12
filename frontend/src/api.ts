// SPDX-FileCopyrightText: © 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * API client for NuRec Web Viewer Backend
 */

import type {
  ScenarioInfo,
  LoadRequest,
  LoadResponse,
  RenderRequest,
  RenderResponse,
  TrajectoryResponse,
  PoseAtTime,
  HealthResponse,
  UsdzFilesResponse,
  NurecRestartRequest,
  NurecRestartResponse,
} from './types';

const API_BASE = '/api';

/**
 * Check backend health status
 */
export async function checkHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE}/health`);
  if (!response.ok) {
    throw new Error(`Health check failed: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Load a USDZ scenario
 */
export async function loadScenario(request: LoadRequest): Promise<LoadResponse> {
  const response = await fetch(`${API_BASE}/load`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || `Load failed: ${response.statusText}`);
  }
  return response.json();
}

/**
 * List USDZ files under backend default sample_set directory
 */
export async function listUsdzFiles(): Promise<UsdzFilesResponse> {
  const response = await fetch(`${API_BASE}/usdz-files`);
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || `Failed to list USDZ files: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Restart NuRec service using selected USDZ path
 */
export async function restartNurecService(request: NurecRestartRequest): Promise<NurecRestartResponse> {
  const response = await fetch(`${API_BASE}/nurec/restart`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || `Failed to restart NuRec: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Get scenario information
 */
export async function getScenarioInfo(): Promise<ScenarioInfo> {
  const response = await fetch(`${API_BASE}/scenario`);
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || `Failed to get scenario: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Get trajectory data
 */
export async function getTrajectory(sampleIntervalUs: number = 100000): Promise<TrajectoryResponse> {
  const response = await fetch(`${API_BASE}/trajectory?sample_interval_us=${sampleIntervalUs}`);
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || `Failed to get trajectory: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Get pose at specific timestamp
 */
export async function getPoseAtTime(timestampUs: number): Promise<PoseAtTime> {
  const response = await fetch(`${API_BASE}/pose?timestamp_us=${timestampUs}`);
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || `Failed to get pose: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Render camera images
 */
export async function renderCameras(request: RenderRequest): Promise<RenderResponse> {
  const response = await fetch(`${API_BASE}/render`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || `Render failed: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Get available camera IDs
 */
export async function getAvailableCameras(): Promise<string[]> {
  const response = await fetch(`${API_BASE}/cameras`);
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || `Failed to get cameras: ${response.statusText}`);
  }
  const data = await response.json();
  return data.cameras;
}

/**
 * Get single camera image URL (for direct img src usage)
 */
export function getCameraImageUrl(cameraId: string, timestampUs: number, scale: number = 0.25): string {
  return `${API_BASE}/render/${cameraId}?timestamp_us=${timestampUs}&scale=${scale}`;
}
