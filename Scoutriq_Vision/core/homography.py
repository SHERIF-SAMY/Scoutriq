"""
homography.py — Perspective transform calibration from known cone positions.

Used for drills with a fixed geometric formation (e.g., diamond drill).
Computes a homography matrix that maps pixel coordinates to real-world metres.

Not used by default / standard drills — only for drills that have a known
cone layout where all positions can be detected.
"""

from __future__ import annotations

import math
import numpy as np
import cv2
from typing import Optional


class HomographyCalibrator:
    """
    Computes a perspective transform (homography) from cone positions
    detected by YOLO on the first frame.

    Diamond drill layout with 5 cones (4m spacing):

                   Top (2, 0)
                  /           \\
    Start (-4, 2) --- Left (0, 2)     Right (4, 2)
                  \\           /
                   Bottom (2, 4)
    """

    CONE_LABELS = ['Top', 'Right', 'Bottom', 'Left', 'Start']

    def __init__(self, cone_spacing: float = 4.0):
        self.cone_spacing = cone_spacing
        self.H: Optional[np.ndarray] = None
        self.detected_pixel_pts: list[tuple[float, float]] = []
        self.labeled_cones: dict[str, tuple[float, float]] = {}

        # Real-world coordinates (metres)
        half = cone_spacing / 2.0
        self.world_pts_5 = np.float32([
            [half, 0.0],                    # Top
            [cone_spacing, half],           # Right
            [half, cone_spacing],           # Bottom
            [0.0, half],                    # Left
            [-cone_spacing, half],          # Start (4m left of Left)
        ])

    def calibrate_from_cones(self, cone_centers: list[tuple[float, float]]) -> Optional[np.ndarray]:
        """
        Given detected cone centers, sort them into positions and compute H.

        Args:
            cone_centers: List of (cx, cy) bounding-box centers for all cones.

        Returns:
            3x3 homography matrix, or None if < 4 cones detected.
        """
        sorted_pts = self._sort_cones_to_positions(cone_centers)
        if len(sorted_pts) < 4:
            return None

        self.detected_pixel_pts = sorted_pts

        # Label the cones
        labels = ['T', 'R', 'B', 'L', 'S']
        for i, label in enumerate(labels[:len(sorted_pts)]):
            self.labeled_cones[label] = sorted_pts[i]

        n = len(sorted_pts)
        pixel_pts = np.float32(sorted_pts)
        world_pts = self.world_pts_5[:n]

        if n == 4:
            self.H = cv2.getPerspectiveTransform(pixel_pts, world_pts)
        else:
            self.H, mask = cv2.findHomography(pixel_pts, world_pts, cv2.RANSAC, 5.0)

        return self.H

    def pixel_to_world(self, px: float, py: float) -> tuple[float, float]:
        """Convert a pixel coordinate to real-world metres using H."""
        if self.H is None:
            raise RuntimeError("Homography not computed")
        pt = np.float64([px, py, 1.0])
        world = self.H @ pt
        return (world[0] / world[2], world[1] / world[2])

    def world_distance(self, p1_px: tuple, p2_px: tuple) -> float:
        """Distance in metres between two pixel points via homography."""
        w1 = self.pixel_to_world(p1_px[0], p1_px[1])
        w2 = self.pixel_to_world(p2_px[0], p2_px[1])
        return math.sqrt((w1[0] - w2[0])**2 + (w1[1] - w2[1])**2)

    @property
    def is_calibrated(self) -> bool:
        return self.H is not None

    def _sort_cones_to_positions(self, centres: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """
        Sort detected cones into [Top, Right, Bottom, Left, Start] order
        based on geometric arrangement.
        """
        n = len(centres)
        if n < 4:
            return []

        pts = sorted(centres, key=lambda p: p[1])  # sort by y

        if n >= 5:
            top = pts[0]
            bottom = pts[-1]
            middle = sorted(pts[1:-1], key=lambda p: p[0])
            if len(middle) >= 3:
                start = middle[0]
                left = middle[1]
                right = middle[2]
                return [top, right, bottom, left, start]
            elif len(middle) >= 2:
                left = middle[0]
                right = middle[1]
                return [top, right, bottom, left]
            return []

        elif n == 4:
            top = pts[0]
            bottom = pts[3]
            middle = sorted(pts[1:3], key=lambda p: p[0])
            left = middle[0]
            right = middle[1]
            return [top, right, bottom, left]

        return []
