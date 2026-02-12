// SPDX-FileCopyrightText: © 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * CameraGrid Component
 * 
 * Displays a draggable grid of camera images from NuRec rendering.
 * Supports layout presets: Grid, Surround, Stack
 */

import React, { useState, useRef, useCallback, useEffect } from 'react';

export type LayoutType = 'grid' | 'surround' | 'stack';

interface CameraPosition {
  x: number;
  y: number;
}

interface CameraSize {
  width: number;
  height: number;
}

interface CameraGridProps {
  selectedCameras: string[];
  images: Record<string, string>;
  currentTimestamp?: number;
}

const CAMERA_LAYOUTS: Record<string, { x: number; y: number; w: number; h: number }> = {
  'front_wide_120fov': { x: 0.35, y: 0.05, w: 0.3, h: 0.4 },
  'front_tele_30fov': { x: 0.35, y: 0.05, w: 0.3, h: 0.4 },
  'left_cross': { x: 0.02, y: 0.25, w: 0.3, h: 0.4 },
  'right_cross': { x: 0.68, y: 0.25, w: 0.3, h: 0.4 },
  'rear_left_70fov': { x: 0.02, y: 0.55, w: 0.3, h: 0.4 },
  'rear_right_70fov': { x: 0.68, y: 0.55, w: 0.3, h: 0.4 },
  'rear_tele_30fov': { x: 0.35, y: 0.55, w: 0.3, h: 0.4 },
};

const formatCameraName = (cameraId: string): string => {
  return cameraId
    .replace('camera_', '')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, l => l.toUpperCase());
};

const CameraGrid: React.FC<CameraGridProps> = ({ selectedCameras, images, currentTimestamp }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [positions, setPositions] = useState<Record<string, CameraPosition>>({});
  const [sizes, setSizes] = useState<Record<string, CameraSize>>({});
  const [dragState, setDragState] = useState<{
    camera: string;
    startX: number;
    startY: number;
    startLeft: number;
    startTop: number;
  } | null>(null);
  const [resizeState, setResizeState] = useState<{
    camera: string;
    startX: number;
    startY: number;
    startWidth: number;
    startHeight: number;
  } | null>(null);
  const [layout, setLayout] = useState<LayoutType>('grid');

  // Load saved layout from localStorage
  useEffect(() => {
    try {
      const saved = localStorage.getItem('nurec-cameraLayout');
      if (saved) {
        const data = JSON.parse(saved);
        if (data.positions) setPositions(data.positions);
        if (data.sizes) setSizes(data.sizes);
      }
    } catch (e) {
      console.warn('Failed to load saved layout:', e);
    }
  }, []);

  // Save layout to localStorage
  const saveLayout = useCallback(() => {
    try {
      localStorage.setItem('nurec-cameraLayout', JSON.stringify({ positions, sizes }));
    } catch (e) {
      console.warn('Failed to save layout:', e);
    }
  }, [positions, sizes]);

  // Get default position for a camera
  const getDefaultPosition = useCallback((_camera: string, index: number): CameraPosition => {
    const container = containerRef.current;
    if (!container) return { x: 15, y: 15 };

    const width = 380;
    const height = 280;
    const gap = 15;
    const cols = Math.min(3, selectedCameras.length);
    const col = index % cols;
    const row = Math.floor(index / cols);

    return {
      x: col * (width + gap) + gap,
      y: row * (height + gap) + gap
    };
  }, [selectedCameras.length]);

  // Apply layout preset
  const applyLayout = useCallback((layoutType: LayoutType) => {
    const container = containerRef.current;
    if (!container) return;

    const containerWidth = container.offsetWidth || 1200;
    const containerHeight = container.offsetHeight || 500;
    const numCameras = selectedCameras.length;

    const newPositions: Record<string, CameraPosition> = {};
    const newSizes: Record<string, CameraSize> = {};

    if (layoutType === 'grid') {
      const cols = Math.min(3, numCameras);
      const width = Math.floor((containerWidth - (cols + 1) * 15) / cols);
      const height = Math.floor(width * 0.6);

      selectedCameras.forEach((camera, index) => {
        const col = index % cols;
        const row = Math.floor(index / cols);
        newPositions[camera] = {
          x: col * (width + 15) + 15,
          y: row * (height + 15) + 15
        };
        newSizes[camera] = { width, height };
      });
    } else if (layoutType === 'surround') {
      selectedCameras.forEach((camera) => {
        const layoutKey = Object.keys(CAMERA_LAYOUTS).find(k => camera.includes(k)) || '';
        const cameraLayout = CAMERA_LAYOUTS[layoutKey] || { x: 0.1, y: 0.1, w: 0.3, h: 0.4 };
        newPositions[camera] = {
          x: Math.floor(cameraLayout.x * containerWidth),
          y: Math.floor(cameraLayout.y * containerHeight)
        };
        newSizes[camera] = {
          width: Math.floor(cameraLayout.w * containerWidth),
          height: Math.floor(cameraLayout.h * containerHeight)
        };
      });
    } else if (layoutType === 'stack') {
      const width = Math.min(600, containerWidth - 30);
      const height = Math.floor((containerHeight - (numCameras + 1) * 10) / numCameras);

      selectedCameras.forEach((camera, index) => {
        newPositions[camera] = {
          x: (containerWidth - width) / 2,
          y: index * (height + 10) + 10
        };
        newSizes[camera] = { width, height };
      });
    }

    setPositions(newPositions);
    setSizes(newSizes);
    setLayout(layoutType);
  }, [selectedCameras]);

  // Reset positions
  const resetPositions = useCallback(() => {
    setPositions({});
    setSizes({});
    localStorage.removeItem('nurec-cameraLayout');
  }, []);

  // Handle mouse down on camera
  const handleMouseDown = useCallback((e: React.MouseEvent, camera: string) => {
    const target = e.target as HTMLElement;
    const cameraView = document.getElementById(`camera-${camera}`);
    if (!cameraView || !containerRef.current) return;

    // Check if clicking resize handle
    if (target.classList.contains('resize-handle')) {
      setResizeState({
        camera,
        startX: e.clientX,
        startY: e.clientY,
        startWidth: cameraView.offsetWidth,
        startHeight: cameraView.offsetHeight
      });
      cameraView.classList.add('dragging');
      e.preventDefault();
      return;
    }

    // Check if clicking drag handle (header)
    if (target.classList.contains('camera-header') || target.closest('.camera-header')) {
      const rect = cameraView.getBoundingClientRect();
      const containerRect = containerRef.current.getBoundingClientRect();

      setDragState({
        camera,
        startX: e.clientX,
        startY: e.clientY,
        startLeft: rect.left - containerRect.left,
        startTop: rect.top - containerRect.top
      });
      cameraView.classList.add('dragging');
      e.preventDefault();
    }
  }, []);

  // Handle mouse move
  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (dragState) {
      const dx = e.clientX - dragState.startX;
      const dy = e.clientY - dragState.startY;
      const newX = Math.max(0, dragState.startLeft + dx);
      const newY = Math.max(0, dragState.startTop + dy);

      setPositions(prev => ({
        ...prev,
        [dragState.camera]: { x: newX, y: newY }
      }));
    }

    if (resizeState) {
      const dx = e.clientX - resizeState.startX;
      const dy = e.clientY - resizeState.startY;
      const newWidth = Math.max(200, resizeState.startWidth + dx);
      const newHeight = Math.max(150, resizeState.startHeight + dy);

      setSizes(prev => ({
        ...prev,
        [resizeState.camera]: { width: newWidth, height: newHeight }
      }));
    }
  }, [dragState, resizeState]);

  // Handle mouse up
  const handleMouseUp = useCallback(() => {
    if (dragState) {
      const cameraView = document.getElementById(`camera-${dragState.camera}`);
      cameraView?.classList.remove('dragging');
      setDragState(null);
      saveLayout();
    }

    if (resizeState) {
      const cameraView = document.getElementById(`camera-${resizeState.camera}`);
      cameraView?.classList.remove('dragging');
      setResizeState(null);
      saveLayout();
    }
  }, [dragState, resizeState, saveLayout]);

  // Add global mouse event listeners
  useEffect(() => {
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [handleMouseMove, handleMouseUp]);

  if (selectedCameras.length === 0) {
    return (
      <>
        <div className="layout-presets">
          <button className="btn btn-settings" disabled>Grid</button>
          <button className="btn btn-settings" disabled>Surround</button>
          <button className="btn btn-settings" disabled>Stack</button>
          <button className="btn btn-settings" disabled>Reset</button>
        </div>
        <div className="video-container" ref={containerRef}>
          <div className="no-content">
            <p>Select cameras below to view</p>
          </div>
        </div>
      </>
    );
  }

  return (
    <>
      {/* Layout Preset Buttons */}
      <div className="layout-presets">
        <button 
          className={`btn btn-settings ${layout === 'grid' ? 'active' : ''}`}
          onClick={() => applyLayout('grid')}
          title="Grid layout"
        >
          Grid
        </button>
        <button 
          className={`btn btn-settings ${layout === 'surround' ? 'active' : ''}`}
          onClick={() => applyLayout('surround')}
          title="Surround view (vehicle perspective)"
        >
          Surround
        </button>
        <button 
          className={`btn btn-settings ${layout === 'stack' ? 'active' : ''}`}
          onClick={() => applyLayout('stack')}
          title="Stacked layout"
        >
          Stack
        </button>
        <button 
          className="btn btn-settings"
          onClick={resetPositions}
          title="Reset to default positions"
        >
          Reset
        </button>
      </div>

      {/* Draggable Camera Container */}
      <div className="video-container" ref={containerRef}>
        <div className="layout-hint">Drag cameras to reposition | Drag corner to resize</div>
        
        {selectedCameras.map((cameraId, index) => {
          const pos = positions[cameraId] || getDefaultPosition(cameraId, index);
          const size = sizes[cameraId] || { width: 380, height: 280 };

          return (
            <div
              key={cameraId}
              id={`camera-${cameraId}`}
              className="camera-view"
              style={{
                left: `${pos.x}px`,
                top: `${pos.y}px`,
                width: `${size.width}px`,
                height: `${size.height}px`,
              }}
              onMouseDown={(e) => handleMouseDown(e, cameraId)}
            >
              <div className="camera-header">
                <div className="camera-name">{formatCameraName(cameraId)}</div>
                {currentTimestamp !== undefined && (
                  <div className="camera-timestamp">
                    TS: {(currentTimestamp / 1_000_000).toFixed(3)}s
                  </div>
                )}
              </div>
              
              {images[cameraId] ? (
                <img
                  src={`data:image/jpeg;base64,${images[cameraId]}`}
                  alt={cameraId}
                />
              ) : (
                <div className="loading">
                  <div className="spinner"></div>
                </div>
              )}
              
              <div className="resize-handle"></div>
            </div>
          );
        })}
      </div>
    </>
  );
};

export default CameraGrid;
