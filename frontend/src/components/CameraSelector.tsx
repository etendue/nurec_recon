// SPDX-FileCopyrightText: © 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * CameraSelector Component
 * 
 * Allows users to select which cameras to display (up to maxCameras).
 */

import React from 'react';
import type { CameraInfo } from '../types';

interface CameraSelectorProps {
  cameras: CameraInfo[];
  selectedCameras: string[];
  onCameraToggle: (cameraId: string) => void;
  maxCameras: number;
}

const CameraSelector: React.FC<CameraSelectorProps> = ({
  cameras,
  selectedCameras,
  onCameraToggle,
  maxCameras,
}) => {
  const isMaxSelected = selectedCameras.length >= maxCameras;

  return (
    <div className="camera-selector">
      <h3>
        Select Cameras ({selectedCameras.length}/{maxCameras})
      </h3>
      <div className="camera-checkbox-list">
        {cameras.map((camera) => {
          const isSelected = selectedCameras.includes(camera.logical_name);
          const isDisabled = !isSelected && isMaxSelected;

          return (
            <div
              key={camera.logical_name}
              className={`camera-checkbox-item ${isDisabled ? 'disabled' : ''}`}
            >
              <input
                type="checkbox"
                id={`cam-${camera.logical_name}`}
                checked={isSelected}
                disabled={isDisabled}
                onChange={() => onCameraToggle(camera.logical_name)}
              />
              <label htmlFor={`cam-${camera.logical_name}`}>
                {camera.logical_name}
                <span style={{ color: '#666', marginLeft: 5, fontSize: '0.8rem' }}>
                  ({camera.resolution[0]}x{camera.resolution[1]})
                </span>
              </label>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default CameraSelector;
