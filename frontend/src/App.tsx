// SPDX-FileCopyrightText: © 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * NuRec Web Viewer - Main Application Component
 * Design consistent with webapp (PhysicalAI-AV Camera Replay)
 */

import React, { useState, useEffect, useCallback } from 'react';
import type { ScenarioInfo } from './types';
import { checkHealth, loadScenario, getScenarioInfo, renderCameras, listUsdzFiles, restartNurecService } from './api';
import CameraGrid from './components/CameraGrid';
import PlaybackControls, { type QualityLevel } from './components/PlaybackControls';
import CameraSelector from './components/CameraSelector';
import { useNuRecPlayback } from './hooks/useNuRecPlayback';

// Resolution scale mapping for quality levels
const QUALITY_SCALES: Record<QualityLevel, number> = {
  low: 0.15,
  medium: 0.25,
  high: 0.5,
};

function App() {
  // Connection state
  const [isConnected, setIsConnected] = useState(false);
  const [scenarioLoaded, setScenarioLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [statusMessage, setStatusMessage] = useState('Ready');

  // Scenario state
  const [scenario, setScenario] = useState<ScenarioInfo | null>(null);
  const [selectedCameras, setSelectedCameras] = useState<string[]>([]);

  // Load form state
  const [usdzPath, setUsdzPath] = useState('');
  const [usdzFiles, setUsdzFiles] = useState<string[]>([]);
  const [nurecHost, setNurecHost] = useState('localhost');
  const [nurecPort, setNurecPort] = useState('46435');

  // Images state
  const [images, setImages] = useState<Record<string, string>>({});

  // Quality state
  const [quality, setQuality] = useState<QualityLevel>('medium');

  // Playback hook
  const {
    playbackState,
    setPlaybackState,
    togglePlay,
    stepForward,
    stepBackward,
    setSpeed,
    seekTo,
  } = useNuRecPlayback(scenario);

  // Check backend health on mount
  useEffect(() => {
    const checkConnection = async () => {
      try {
        const health = await checkHealth();
        setIsConnected(health.grpc_connected);
        setScenarioLoaded(health.scenario_loaded);
        if (!health.grpc_connected) {
          setStatusMessage('NuRec disconnected');
        }

        // If scenario is already loaded, fetch info
        if (health.scenario_loaded) {
          const info = await getScenarioInfo();
          setScenario(info);
          // Keep user's current camera selection if still valid; otherwise fall back.
          setSelectedCameras(prev => {
            const available = new Set(info.cameras.map(c => c.logical_name));
            const validPrev = prev.filter(id => available.has(id));
            if (validPrev.length > 0) {
              return validPrev;
            }
            return info.cameras.slice(0, 3).map(c => c.logical_name);
          });
          setStatusMessage('Scenario loaded');
        }
      } catch (err) {
        console.error('Health check failed:', err);
        setIsConnected(false);
        setStatusMessage('Backend disconnected');
      }
    };

    checkConnection();
    // Periodically check connection
    const interval = setInterval(checkConnection, 10000);
    return () => clearInterval(interval);
  }, []);

  // Load USDZ options from default sample_set
  useEffect(() => {
    const fetchUsdzFiles = async () => {
      try {
        const response = await listUsdzFiles();
        setUsdzFiles(response.files);
        setUsdzPath((prev) => prev || response.files[0] || '');
      } catch (err) {
        console.error('Failed to list USDZ files:', err);
      }
    };
    fetchUsdzFiles();
  }, []);

  // Handle load scenario
  const handleLoadScenario = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setIsLoading(true);
    setStatusMessage('Loading scenario...');

    try {
      setStatusMessage('Restarting NuRec service...');
      const restartResp = await restartNurecService({
        usdz_path: usdzPath,
        nurec_host: nurecHost,
        nurec_port: parseInt(nurecPort, 10),
      });
      if (!restartResp.grpc_ready) {
        throw new Error('NuRec service is still starting. Please retry in a few seconds.');
      }

      setStatusMessage('Loading scenario...');
      await loadScenario({
        usdz_path: usdzPath,
        nurec_host: nurecHost,
        nurec_port: parseInt(nurecPort, 10),
      });

      const info = await getScenarioInfo();
      setScenario(info);
      setScenarioLoaded(true);

      // Select first 3 cameras by default
      const defaultCameras = info.cameras.slice(0, 3).map(c => c.logical_name);
      setSelectedCameras(defaultCameras);

      // Reset playback state
      setPlaybackState({
        isPlaying: false,
        currentTime_us: info.time_range.start_us,
        playbackSpeed: 1,
      });

      setStatusMessage(`Loaded: ${info.sequence_id}`);
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : 'Failed to load scenario';
      setError(errMsg);
      setStatusMessage(`Error: ${errMsg}`);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle camera selection change
  const handleCameraToggle = useCallback((cameraId: string) => {
    setSelectedCameras(prev => {
      if (prev.includes(cameraId)) {
        return prev.filter(c => c !== cameraId);
      } else if (prev.length < 3) {
        return [...prev, cameraId];
      }
      return prev;
    });
  }, []);

  // Handle quality change
  const handleQualityChange = useCallback((newQuality: QualityLevel) => {
    setQuality(newQuality);
    setStatusMessage(`Quality set to ${newQuality}`);
  }, []);

  // Fetch images when time or cameras change
  useEffect(() => {
    if (!scenarioLoaded || selectedCameras.length === 0) return;

    const fetchImages = async () => {
      try {
        const response = await renderCameras({
          timestamp_us: Math.floor(playbackState.currentTime_us),
          camera_ids: selectedCameras,
          resolution_scale: QUALITY_SCALES[quality],
        });
        setImages(response.images);
      } catch (err) {
        console.error('Render failed:', err);
        setStatusMessage('Render error');
      }
    };

    fetchImages();
  }, [playbackState.currentTime_us, selectedCameras, scenarioLoaded, quality]);

  // Format time display
  const formatTime = (us: number, startUs: number = 0): string => {
    const seconds = (us - startUs) / 1_000_000;
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${String(mins).padStart(2, '0')}:${secs.toFixed(3).padStart(6, '0')}`;
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <h1>NVIDIA PhysicalAI - NuRec Web Viewer</h1>
        <div className="status-indicator">
          <span className={`status-dot ${isConnected ? 'connected' : 'disconnected'}`} />
          <span>{isConnected ? 'NuRec Connected' : 'NuRec Disconnected'}</span>
        </div>
      </header>

      {/* Error Message */}
      {error && (
        <div className="error-message">
          {error}
          <button onClick={() => setError(null)} style={{ marginLeft: 15 }}>Dismiss</button>
        </div>
      )}

      {/* Load Panel */}
      <div className="load-panel">
        <h2>Load USDZ Scenario</h2>
        <form className="load-form" onSubmit={handleLoadScenario}>
          <select
            value={usdzPath}
            onChange={(e) => setUsdzPath(e.target.value)}
            style={{ minWidth: 420, flex: 1 }}
            required
          >
            {usdzFiles.length === 0 ? (
              <option value="">No USDZ files found in sample_set</option>
            ) : (
              usdzFiles.map((path) => (
                <option key={path} value={path}>{path}</option>
              ))
            )}
          </select>
          <input
            type="text"
            placeholder="NuRec Host"
            value={nurecHost}
            onChange={(e) => setNurecHost(e.target.value)}
            style={{ minWidth: 120, flex: 0 }}
          />
          <input
            type="text"
            placeholder="Port"
            value={nurecPort}
            onChange={(e) => setNurecPort(e.target.value)}
            style={{ minWidth: 80, flex: 0 }}
          />
          <button type="submit" disabled={isLoading || !usdzPath}>
            {isLoading ? 'Loading...' : 'Load'}
          </button>
        </form>
      </div>

      {/* Main Content (show when scenario loaded) */}
      {scenarioLoaded && scenario && (
        <>
          {/* Clip Info Bar */}
          <div className="clip-info">
            <span>Scenario: {scenario.sequence_id}</span>
            <span>Duration: {scenario.time_range.duration_seconds.toFixed(1)}s</span>
            <span>Cameras: {scenario.cameras.length} available</span>
          </div>

          {/* Camera Grid with Layout Presets */}
          <CameraGrid
            selectedCameras={selectedCameras}
            images={images}
            currentTimestamp={playbackState.currentTime_us}
          />

          {/* Playback Controls with Quality */}
          <PlaybackControls
            playbackState={playbackState}
            timeRange={scenario.time_range}
            onTogglePlay={togglePlay}
            onStepForward={stepForward}
            onStepBackward={stepBackward}
            onSeek={seekTo}
            onSpeedChange={setSpeed}
            onQualityChange={handleQualityChange}
            quality={quality}
            formatTime={(us) => formatTime(us, scenario.time_range.start_us)}
          />

          {/* Camera Selector */}
          <CameraSelector
            cameras={scenario.cameras}
            selectedCameras={selectedCameras}
            onCameraToggle={handleCameraToggle}
            maxCameras={3}
          />
        </>
      )}

      {/* Loading Spinner */}
      {isLoading && (
        <div className="loading">
          <div className="spinner" />
        </div>
      )}

      {/* Status Bar */}
      <div className="status-bar">
        <span id="status-message">{statusMessage}</span>
        <span className="decoder-info">NuRec gRPC Renderer</span>
      </div>
    </div>
  );
}

export default App;
