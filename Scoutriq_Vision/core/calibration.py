"""
calibration.py — Pixel-to-metre calibration from known physical references.

Consolidates the calibration logic previously repeated with small variations in
diamond.py, seven_cone_dribble.py, weakfoot_ultimate.py, weekfootgpt.py.

Two strategies:
  1. PlayerCalibrator — uses head-to-ankle pixel height vs known player height.
  2. BallCalibrator — uses football bounding-box diameter vs 21.5 cm standard.

PlayerCalibrator is preferred (more stable); BallCalibrator is the fallback.
"""

from __future__ import annotations

from typing import Optional
import numpy as np

from ..constants import (
    KP_NOSE,
    KP_L_ANKLE,
    KP_R_ANKLE,
    DEFAULT_PLAYER_HEIGHT_M,
    DEFAULT_FOOTBALL_DIAMETER_M,
)


class PlayerCalibrator:
    """
    Calibrate pixels_per_meter from the player's height in pixels.

    Uses the vertical distance from the nose (keypoint 0) to the average
    ankle position (keypoints 15/16) compared to a known player height.

    Usage:
        cal = PlayerCalibrator(player_height_m=1.75)
        # each frame:
        cal.update(keypoints, conf_threshold=0.3)
        # after processing (or periodically):
        ppm = cal.pixels_per_meter  # may be None if not enough samples
    """

    def __init__(self, player_height_m: float = DEFAULT_PLAYER_HEIGHT_M):
        self.player_height_m = player_height_m
        self._heights_px: list[float] = []
        self._pixels_per_meter: Optional[float] = None

    def update(self, keypoints: np.ndarray, conf_threshold: float = 0.3) -> None:
        """
        Collect a height sample from a single frame's keypoints.

        Args:
            keypoints: Shape (N, 3) — [x, y, conf] per keypoint (COCO-17).
            conf_threshold: Minimum confidence to accept a keypoint.
        """
        if len(keypoints) < 17:
            return

        nose = keypoints[KP_NOSE]
        l_ankle = keypoints[KP_L_ANKLE]
        r_ankle = keypoints[KP_R_ANKLE]

        if nose[2] < conf_threshold:
            return

        # Pick the best ankle(s)
        ankle_y: Optional[float] = None
        if l_ankle[2] > conf_threshold and r_ankle[2] > conf_threshold:
            ankle_y = (l_ankle[1] + r_ankle[1]) / 2.0
        elif l_ankle[2] > conf_threshold:
            ankle_y = float(l_ankle[1])
        elif r_ankle[2] > conf_threshold:
            ankle_y = float(r_ankle[1])

        if ankle_y is None:
            return

        height_px = abs(ankle_y - float(nose[1]))
        if height_px > 50:  # reject noise / crouching
            self._heights_px.append(height_px)

    def compute(self) -> Optional[float]:
        """
        Calculate pixels_per_meter from collected samples.

        Uses the *maximum* observed height (player standing upright).
        """
        if not self._heights_px:
            return None
        max_h = max(self._heights_px)
        self._pixels_per_meter = max_h / self.player_height_m
        return self._pixels_per_meter

    @property
    def pixels_per_meter(self) -> Optional[float]:
        """Current calibration value, or None if not yet computed."""
        return self._pixels_per_meter

    @property
    def sample_count(self) -> int:
        return len(self._heights_px)


class BallCalibrator:
    """
    Fallback calibration using the known diameter of a standard football.

    Collects max-dimension measurements from ball bounding boxes and
    derives pixels_per_meter = max_ball_px / football_diameter_m.
    """

    def __init__(self, football_diameter_m: float = DEFAULT_FOOTBALL_DIAMETER_M):
        self.football_diameter_m = football_diameter_m
        self._sizes_px: list[float] = []
        self._pixels_per_meter: Optional[float] = None

    def update(self, box) -> None:
        """
        Collect a ball-size sample from a bounding box.

        Args:
            box: (x1, y1, x2, y2) bounding box.
        """
        x1, y1, x2, y2 = box
        size = max(x2 - x1, y2 - y1)
        if size > 10:
            self._sizes_px.append(float(size))

    def compute(self) -> Optional[float]:
        """
        Calculate pixels_per_meter from the maximum observed ball size.
        """
        if not self._sizes_px:
            return None
        max_ball_px = max(self._sizes_px)
        self._pixels_per_meter = max_ball_px / self.football_diameter_m
        return self._pixels_per_meter

    @property
    def pixels_per_meter(self) -> Optional[float]:
        return self._pixels_per_meter

    @property
    def sample_count(self) -> int:
        return len(self._sizes_px)


class CalibrationManager:
    """
    Orchestrates player-first, ball-fallback calibration.

    Provides a single `px_to_m(pixels)` method that works regardless
    of which calibrator succeeded.
    """

    def __init__(
        self,
        player_height_m: float = DEFAULT_PLAYER_HEIGHT_M,
        football_diameter_m: float = DEFAULT_FOOTBALL_DIAMETER_M,
    ):
        self.player = PlayerCalibrator(player_height_m)
        self.ball = BallCalibrator(football_diameter_m)
        self._pixels_per_meter: Optional[float] = None
        self._source: str = "none"

    def update_from_keypoints(
        self, keypoints: np.ndarray, conf_threshold: float = 0.3
    ) -> None:
        """Feed keypoints for player-height calibration."""
        self.player.update(keypoints, conf_threshold)

    def update_from_ball_box(self, box) -> None:
        """Feed a ball bounding box for fallback calibration."""
        self.ball.update(box)

    def compute(self) -> Optional[float]:
        """
        Compute the best available calibration — called every frame.

        Priority: ball diameter (primary) > player height (fallback).
        Ball is preferred because it's more consistent across frames.
        Player height is fallback for drills without a ball.
        """
        # Primary: ball diameter
        ppm = self.ball.compute()
        if ppm is not None:
            self._pixels_per_meter = ppm
            self._source = "ball_diameter"
            return ppm

        # Fallback: player height
        ppm = self.player.compute()
        if ppm is not None:
            self._pixels_per_meter = ppm
            self._source = "player_height"
            return ppm

        return None

    @property
    def is_calibrated(self) -> bool:
        ppm = self._pixels_per_meter
        return ppm is not None and ppm > 0

    @property
    def pixels_per_meter(self) -> Optional[float]:
        return self._pixels_per_meter

    @property
    def source(self) -> str:
        return self._source

    def px_to_m(self, px: float) -> float:
        """Convert pixel distance to metres. Returns raw px / 250 if uncalibrated."""
        ppm = self._pixels_per_meter
        if ppm is not None and ppm > 0:
            return float(px / ppm)
        return px / 250.0 # Default scale fallback

    def m_to_px(self, m: float) -> float:
        """Convert metric distance to pixels. Returns m * 250 if uncalibrated."""
        ppm = self._pixels_per_meter
        if ppm is not None and ppm > 0:
            return float(m * ppm)
        return m * 250.0 # Default scale fallback
