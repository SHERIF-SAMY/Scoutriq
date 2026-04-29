"""
ball_physics.py — Physics-based ball velocity tracking.

Provides the BallVelocityTracker class that computes instantaneous
velocity, velocity deltas (spikes), and direction changes for the ball.

Used by drill analyzers to distinguish real touches (foot + velocity
spike) from false positives (foot near ball but no physical contact).

All computations are O(1) per frame — pure math, zero GPU cost.
"""

from __future__ import annotations

import math
from typing import Optional


class BallVelocityTracker:
    """
    Tracks ball position across frames and computes physics metrics.

    Usage in a drill:
        tracker = BallVelocityTracker()

        # Every frame:
        tracker.update(frame_num, ball_position)  # or None if ball not seen

        # Query:
        tracker.velocity          → current speed (px/frame)
        tracker.velocity_delta    → change in speed since last frame
        tracker.has_velocity_spike(threshold) → True if Δv > threshold

    Args:
        history_size: Number of past positions to keep for smoothed
                      velocity computation. Default 5.
    """

    def __init__(self, history_size: int = 5):
        self._history_size = history_size

        # State
        self._positions: list[tuple[int, float, float]] = []  # (frame, x, y)
        self._velocities: list[float] = []                     # speed history
        self._directions: list[float] = []                     # angle history (radians)

        # Public read-only
        self.velocity: float = 0.0          # current speed (px/frame)
        self.velocity_delta: float = 0.0    # Δ speed since last update
        self.direction: float = 0.0         # current direction (radians)
        self.direction_delta: float = 0.0   # Δ direction since last update (radians)

    def update(self, frame_num: int, position: Optional[tuple[float, float]]) -> None:
        """
        Record ball position for this frame.

        Args:
            frame_num: Current frame number.
            position:  (x, y) of ball center, or None if ball not detected.
        """
        if position is None:
            return

        x, y = float(position[0]), float(position[1])
        self._positions.append((frame_num, x, y))

        # Trim history
        if len(self._positions) > self._history_size * 2:
            self._positions = self._positions[-self._history_size * 2:]

        # Need at least 2 positions to compute velocity
        if len(self._positions) < 2:
            return

        prev_frame, px, py = self._positions[-2]
        curr_frame, cx, cy = self._positions[-1]

        dt = curr_frame - prev_frame
        if dt <= 0 or dt > 5:
            # Non-consecutive or too far apart — reset velocity
            self.velocity = 0.0
            self.velocity_delta = 0.0
            return

        dx = cx - px
        dy = cy - py
        dist = math.sqrt(dx * dx + dy * dy)
        speed = dist / dt  # px per frame

        # Direction (angle in radians)
        direction = math.atan2(dy, dx) if dist > 1e-3 else self.direction

        # Velocity delta
        prev_vel = self.velocity
        self.velocity = speed
        self.velocity_delta = abs(speed - prev_vel)

        # Direction delta (handle wraparound)
        prev_dir = self.direction
        self.direction = direction
        raw_delta = direction - prev_dir
        # Normalize to [-π, π]
        self.direction_delta = abs(math.atan2(math.sin(raw_delta), math.cos(raw_delta)))

        # Keep history
        self._velocities.append(speed)
        self._directions.append(direction)
        if len(self._velocities) > self._history_size:
            self._velocities = self._velocities[-self._history_size:]
        if len(self._directions) > self._history_size:
            self._directions = self._directions[-self._history_size:]

    def has_velocity_spike(self, threshold: float = 3.0) -> bool:
        """
        Check if the ball experienced a sudden velocity change.

        Args:
            threshold: Minimum Δv (px/frame) to count as a spike.

        Returns:
            True if the velocity changed by more than ``threshold``
            since the last frame.
        """
        return self.velocity_delta > threshold

    def has_direction_change(self, threshold_deg: float = 30.0) -> bool:
        """
        Check if the ball abruptly changed direction.

        Args:
            threshold_deg: Minimum direction change in degrees.

        Returns:
            True if direction changed by more than ``threshold_deg``.
        """
        return self.direction_delta > math.radians(threshold_deg)

    def has_physical_touch(
        self,
        velocity_threshold: float = 3.0,
        direction_threshold_deg: float = 30.0,
    ) -> bool:
        """
        Combined check: did a physical interaction (touch) likely occur?

        A touch is detected if EITHER:
          - velocity spiked (ball sped up or slowed down suddenly), OR
          - direction changed abruptly

        This is meant to be combined with a proximity check in the drill.

        Args:
            velocity_threshold:     Minimum Δv (px/frame).
            direction_threshold_deg: Minimum direction change (degrees).

        Returns:
            True if physics suggest a touch happened this frame.
        """
        return (
            self.has_velocity_spike(velocity_threshold)
            or self.has_direction_change(direction_threshold_deg)
        )

    def get_smoothed_velocity(self) -> float:
        """Average velocity over the recent history window."""
        if not self._velocities:
            return 0.0
        return sum(self._velocities) / len(self._velocities)

    def reset(self) -> None:
        """Clear all tracking state."""
        self._positions.clear()
        self._velocities.clear()
        self._directions.clear()
        self.velocity = 0.0
        self.velocity_delta = 0.0
        self.direction = 0.0
        self.direction_delta = 0.0
