"""
waypoint_tracker.py — Track player progress through a fixed cone journey.

Used for drills with a predetermined path (e.g., diamond: S→L→B→R→T→L→S).
Computes per-leg speed using known inter-cone distance and transit time,
which is far more accurate than pixel-displacement speed.
"""

from __future__ import annotations

import math
from typing import Optional


class WaypointTracker:
    """
    Tracks the player's progress through a cone drill by detecting
    proximity to each waypoint cone.

    Speed = cone_spacing / transit_time (known distance, measured time).
    """

    def __init__(
        self,
        journey: list[str],
        cone_pixel_positions: dict[str, tuple[float, float]],
        cone_spacing: float,
        fps: float,
        proximity_ratio: float = 0.25,
    ):
        """
        Args:
            journey: Cone labels in visit order, e.g. ['S','L','B','R','T','L','S']
            cone_pixel_positions: dict mapping label → (cx, cy) pixel position
            cone_spacing: Real-world distance between adjacent cones (metres)
            fps: Video frame rate
            proximity_ratio: Fraction of avg inter-cone pixel distance used as
                             "arrival" threshold (0.25 = within 25% of gap)
        """
        self.journey = journey
        self.cone_px = cone_pixel_positions
        self.cone_spacing = cone_spacing
        self.fps = fps

        self.current_target_idx: int = 0
        self.completed_legs: list[dict] = []
        self._leg_start_frame: int = 0

        # Compute arrival threshold in pixels
        dists = []
        for i in range(len(journey) - 1):
            a, b = journey[i], journey[i + 1]
            if a in self.cone_px and b in self.cone_px:
                pa, pb = self.cone_px[a], self.cone_px[b]
                d = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
                dists.append(d)
        self.avg_cone_dist_px = float(sum(dists) / len(dists)) if dists else 100.0
        self.proximity_threshold = self.avg_cone_dist_px * proximity_ratio

    def update(self, player_pos: tuple[float, float], frame_idx: int):
        """Check if the player has reached the next waypoint cone."""
        if self.current_target_idx >= len(self.journey):
            return  # drill completed

        target_label = self.journey[self.current_target_idx]
        if target_label not in self.cone_px:
            return

        target_pos = self.cone_px[target_label]
        dist = math.sqrt(
            (player_pos[0] - target_pos[0])**2 +
            (player_pos[1] - target_pos[1])**2
        )

        if dist < self.proximity_threshold:
            if self.current_target_idx == 0:
                # First waypoint → start timing
                self._leg_start_frame = frame_idx
            else:
                # Completed a leg
                frames_elapsed = frame_idx - self._leg_start_frame
                time_s = frames_elapsed / self.fps if self.fps > 0 else 1.0
                speed_mps = self.cone_spacing / time_s if time_s > 0 else 0.0
                from_label = self.journey[self.current_target_idx - 1]

                self.completed_legs.append({
                    'from': from_label,
                    'to': target_label,
                    'start_frame': self._leg_start_frame,
                    'end_frame': frame_idx,
                    'time_seconds': round(time_s, 3),
                    'speed_mps': round(speed_mps, 2),
                    'speed_kmh': round(speed_mps * 3.6, 2),
                })
                self._leg_start_frame = frame_idx

            self.current_target_idx += 1

    def current_target(self) -> Optional[str]:
        """Label of the next target cone, or None if drill is done."""
        if self.current_target_idx < len(self.journey):
            return self.journey[self.current_target_idx]
        return None

    @property
    def is_complete(self) -> bool:
        return self.current_target_idx >= len(self.journey)

    @property
    def legs_completed(self) -> int:
        return len(self.completed_legs)

    @property
    def total_distance(self) -> float:
        """Total distance covered = completed legs × cone_spacing."""
        return len(self.completed_legs) * self.cone_spacing

    def summary(self) -> dict:
        """Return summary statistics."""
        if not self.completed_legs:
            return {
                'total_distance_meters': 0.0,
                'total_time_seconds': 0.0,
                'average_speed_mps': 0.0,
                'average_speed_kmh': 0.0,
                'max_speed_mps': 0.0,
                'max_speed_kmh': 0.0,
                'legs_completed': 0,
                'legs_expected': len(self.journey) - 1,
                'legs': [],
            }

        total_dist = self.total_distance
        total_time = sum(leg['time_seconds'] for leg in self.completed_legs)
        avg_mps = total_dist / total_time if total_time > 0 else 0.0
        max_mps = max(leg['speed_mps'] for leg in self.completed_legs)

        return {
            'total_distance_meters': round(total_dist, 2),
            'total_time_seconds': round(total_time, 2),
            'average_speed_mps': round(avg_mps, 2),
            'average_speed_kmh': round(avg_mps * 3.6, 2),
            'max_speed_mps': round(max_mps, 2),
            'max_speed_kmh': round(max_mps * 3.6, 2),
            'legs_completed': len(self.completed_legs),
            'legs_expected': len(self.journey) - 1,
            'legs': self.completed_legs,
        }
