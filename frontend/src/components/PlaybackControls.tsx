// SPDX-FileCopyrightText: © 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * PlaybackControls Component
 * 
 * Provides play/pause, forward/backward, timeline slider, speed, and quality controls.
 */

import React from 'react';
import type { PlaybackState, TimeRange } from '../types';

export type QualityLevel = 'low' | 'medium' | 'high';

interface PlaybackControlsProps {
  playbackState: PlaybackState;
  timeRange: TimeRange;
  onTogglePlay: () => void;
  onStepForward: (seconds: number) => void;
  onStepBackward: (seconds: number) => void;
  onSeek: (timestampUs: number) => void;
  onSpeedChange: (speed: number) => void;
  onQualityChange?: (quality: QualityLevel) => void;
  quality?: QualityLevel;
  formatTime: (us: number) => string;
}

const PlaybackControls: React.FC<PlaybackControlsProps> = ({
  playbackState,
  timeRange,
  onTogglePlay,
  onStepForward,
  onStepBackward,
  onSeek,
  onSpeedChange,
  onQualityChange,
  quality = 'medium',
  formatTime,
}) => {
  const speeds = [0.5, 1, 2];

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value, 10);
    onSeek(value);
  };

  const handleQualityChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    if (onQualityChange) {
      onQualityChange(e.target.value as QualityLevel);
    }
  };

  return (
    <div className="playback-controls">
      {/* Timeline Slider */}
      <div className="timeline-container">
        <input
          type="range"
          className="timeline-slider"
          min={timeRange.start_us}
          max={timeRange.end_us}
          value={playbackState.currentTime_us}
          onChange={handleSliderChange}
        />
        <div className="time-display">
          <span>{formatTime(playbackState.currentTime_us)}</span>
          <span>/</span>
          <span>{formatTime(timeRange.end_us)}</span>
        </div>
      </div>

      {/* Playback Buttons */}
      <div className="playback-buttons">
        <button className="playback-btn" onClick={() => onStepBackward(5)} title="Backward 5s">
          ⏪
        </button>
        <button className="playback-btn" onClick={() => onStepBackward(1)} title="Backward 1s">
          ◀
        </button>
        <button className="playback-btn play-pause" onClick={onTogglePlay} title="Play/Pause">
          {playbackState.isPlaying ? '⏸' : '▶'}
        </button>
        <button className="playback-btn" onClick={() => onStepForward(1)} title="Forward 1s">
          ▶
        </button>
        <button className="playback-btn" onClick={() => onStepForward(5)} title="Forward 5s">
          ⏩
        </button>

        {/* Speed Control */}
        <div className="speed-control">
          <label>Speed:</label>
          {speeds.map((speed) => (
            <button
              key={speed}
              className={`speed-btn ${playbackState.playbackSpeed === speed ? 'active' : ''}`}
              onClick={() => onSpeedChange(speed)}
            >
              {speed}x
            </button>
          ))}
        </div>

        {/* Quality Control */}
        {onQualityChange && (
          <div className="quality-control">
            <label>Quality:</label>
            <select value={quality} onChange={handleQualityChange}>
              <option value="low">Low (Fast)</option>
              <option value="medium">Medium</option>
              <option value="high">High (Slow)</option>
            </select>
          </div>
        )}
      </div>
    </div>
  );
};

export default PlaybackControls;
