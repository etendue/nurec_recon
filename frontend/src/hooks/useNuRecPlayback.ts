// SPDX-FileCopyrightText: © 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * useNuRecPlayback Hook
 * 
 * Manages playback state and provides control functions for the NuRec viewer.
 */

import { useState, useCallback, useEffect, useRef } from 'react';
import type { ScenarioInfo, PlaybackState } from '../types';

interface UseNuRecPlaybackReturn {
  playbackState: PlaybackState;
  setPlaybackState: React.Dispatch<React.SetStateAction<PlaybackState>>;
  togglePlay: () => void;
  stepForward: (seconds: number) => void;
  stepBackward: (seconds: number) => void;
  setSpeed: (speed: number) => void;
  seekTo: (timestampUs: number) => void;
}

export function useNuRecPlayback(scenario: ScenarioInfo | null): UseNuRecPlaybackReturn {
  const [playbackState, setPlaybackState] = useState<PlaybackState>({
    isPlaying: false,
    currentTime_us: 0,
    playbackSpeed: 1,
  });

  const animationRef = useRef<number>();
  const lastFrameTimeRef = useRef<number>(0);

  // Initialize playback state when scenario loads
  useEffect(() => {
    if (scenario) {
      setPlaybackState((prev) => ({
        ...prev,
        currentTime_us: scenario.time_range.start_us,
      }));
    }
  }, [scenario]);

  // Animation loop for playback
  useEffect(() => {
    if (!playbackState.isPlaying || !scenario) {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      return;
    }

    const targetFPS = 10; // Target FPS for rendering (limited by NuRec render speed)
    const frameInterval = 1000 / targetFPS;

    const animate = (timestamp: number) => {
      if (timestamp - lastFrameTimeRef.current >= frameInterval) {
        const deltaMs = timestamp - lastFrameTimeRef.current;
        lastFrameTimeRef.current = timestamp;

        // Calculate time step in microseconds
        // Real-time playback: 1 second of real time = 1 second of scenario time
        const timeStep_us = deltaMs * 1000 * playbackState.playbackSpeed;

        setPlaybackState((prev) => {
          const newTime = prev.currentTime_us + timeStep_us;

          // Check if we've reached the end
          if (newTime >= scenario.time_range.end_us) {
            return {
              ...prev,
              isPlaying: false,
              currentTime_us: scenario.time_range.end_us,
            };
          }

          return {
            ...prev,
            currentTime_us: newTime,
          };
        });
      }

      animationRef.current = requestAnimationFrame(animate);
    };

    lastFrameTimeRef.current = performance.now();
    animationRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [playbackState.isPlaying, playbackState.playbackSpeed, scenario]);

  // Toggle play/pause
  const togglePlay = useCallback(() => {
    setPlaybackState((prev) => {
      // Only auto-restart when user presses Play while already paused at end.
      if (scenario && !prev.isPlaying && prev.currentTime_us >= scenario.time_range.end_us) {
        return {
          ...prev,
          isPlaying: true,
          currentTime_us: scenario.time_range.start_us,
        };
      }
      return {
        ...prev,
        isPlaying: !prev.isPlaying,
      };
    });
  }, [scenario]);

  // Step forward by seconds
  const stepForward = useCallback(
    (seconds: number) => {
      if (!scenario) return;

      setPlaybackState((prev) => ({
        ...prev,
        isPlaying: false,
        currentTime_us: Math.min(
          prev.currentTime_us + seconds * 1_000_000,
          scenario.time_range.end_us
        ),
      }));
    },
    [scenario]
  );

  // Step backward by seconds
  const stepBackward = useCallback(
    (seconds: number) => {
      if (!scenario) return;

      setPlaybackState((prev) => ({
        ...prev,
        isPlaying: false,
        currentTime_us: Math.max(
          prev.currentTime_us - seconds * 1_000_000,
          scenario.time_range.start_us
        ),
      }));
    },
    [scenario]
  );

  // Set playback speed
  const setSpeed = useCallback((speed: number) => {
    setPlaybackState((prev) => ({
      ...prev,
      playbackSpeed: speed,
    }));
  }, []);

  // Seek to specific timestamp
  const seekTo = useCallback(
    (timestampUs: number) => {
      if (!scenario) return;

      setPlaybackState((prev) => ({
        ...prev,
        isPlaying: false,
        currentTime_us: Math.max(
          scenario.time_range.start_us,
          Math.min(timestampUs, scenario.time_range.end_us)
        ),
      }));
    },
    [scenario]
  );

  return {
    playbackState,
    setPlaybackState,
    togglePlay,
    stepForward,
    stepBackward,
    setSpeed,
    seekTo,
  };
}
