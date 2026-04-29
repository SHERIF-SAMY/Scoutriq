"""
stable_tracker.py — Position-based stable ID assignment for tracked objects.

Previously duplicated (identically) in:
  diamond.py, seven_cone_dribble.py

When the YOLO tracker's internal ID changes (e.g., object temporarily lost),
this module re-associates the new tracker ID with the original stable ID
based on spatial proximity, giving consistent IDs across the full video.
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict

from .geometry import box_center, dist2d


class StableIDTracker:
    """
    Maps transient YOLO tracker IDs to spatially-stable IDs.

    Args:
        position_threshold: Maximum pixel distance to consider
            a new detection as the same object.
    """

    def __init__(self, position_threshold: float = 50.0):
        self.position_threshold = position_threshold

        # {(stable_id, class_id): ((cx, cy), class_id)}
        self.id_positions: dict[tuple[int, int], tuple[tuple[float, float], int]] = {}

        # {(tracker_id, class_id): (stable_id, class_id)}
        self.tracker_to_stable: dict[tuple[int, int], tuple[int, int]] = {}

        # Next stable ID to assign per class
        self.next_id_per_class: dict[int, int] = defaultdict(lambda: 1)

        # IDs seen this frame (for end_frame cleanup)
        self.active_ids: set[tuple[int, int]] = set()

    # ── helpers ──

    @staticmethod
    def _make_key(tracker_id: int, class_id: int) -> tuple[int, int]:
        return (int(tracker_id), int(class_id))

    @staticmethod
    def _make_stable_key(stable_id: int, class_id: int) -> tuple[int, int]:
        return (int(stable_id), int(class_id))

    def _find_matching_stable_id(
        self, center: tuple[float, float], class_id: int
    ) -> tuple[int, int] | None:
        """Find the closest inactive stable ID of the same class."""
        best_match = None
        best_distance = float("inf")
        for stable_key, (pos, cls) in self.id_positions.items():
            if cls != class_id:
                continue
            d = dist2d(center, pos)
            if d < self.position_threshold and d < best_distance:
                if stable_key not in self.active_ids:
                    best_match = stable_key
                    best_distance = d
        return best_match

    # ── public API ──

    def update(self, tracker_id: int, box, class_id: int) -> int:
        """
        Register a detection and return its stable ID.

        Args:
            tracker_id: The YOLO tracker's transient ID.
            box: Bounding box (x1, y1, x2, y2).
            class_id: Detected class index.

        Returns:
            Stable ID (int) that is consistent across frames.
        """
        center = box_center(box)
        key = self._make_key(tracker_id, class_id)

        # 1. Already mapped from a previous frame
        if key in self.tracker_to_stable:
            stable_key = self.tracker_to_stable[key]
            self.id_positions[stable_key] = (center, class_id)
            self.active_ids.add(stable_key)
            return stable_key[0]

        # 2. Match to an existing stable ID by proximity
        matching = self._find_matching_stable_id(center, class_id)
        if matching is not None:
            self.tracker_to_stable[key] = matching
            self.id_positions[matching] = (center, class_id)
            self.active_ids.add(matching)
            return matching[0]

        # 3. Assign a new stable ID
        stable_id = self.next_id_per_class[class_id]
        self.next_id_per_class[class_id] += 1
        stable_key = self._make_stable_key(stable_id, class_id)
        self.tracker_to_stable[key] = stable_key
        self.id_positions[stable_key] = (center, class_id)
        self.active_ids.add(stable_key)
        return stable_id

    def end_frame(self) -> None:
        """Call at the end of each frame to reset the active-ID set."""
        self.active_ids.clear()

    def reset(self) -> None:
        """Clear all tracking state."""
        self.id_positions.clear()
        self.tracker_to_stable.clear()
        self.next_id_per_class.clear()
        self.active_ids.clear()
