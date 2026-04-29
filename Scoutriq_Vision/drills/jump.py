"""
jump.py — Vertical Jump / CMJ Analyzer.

Analyzes a countermovement jump (CMJ) or vertical jump from video.
The CORE purpose of this drill is to verify the player keeps legs
straight during the jump — any knee bend in the air is flagged.

Measurements:
  • Jump height (ankle displacement, in cm/m)
  • Knee angles (L + R) per frame
  • Knee bend detection during airtime (< 160° = bent)
  • Form assessment (straight legs = good form)
"""

from __future__ import annotations

import math
import numpy as np
from typing import Optional

from ..base_drill import BaseDrillAnalyzer
from ..config import DrillConfig
from ..constants import (
    KP_L_HIP, KP_R_HIP, KP_L_KNEE, KP_R_KNEE,
    KP_L_ANKLE, KP_R_ANKLE, COLOR_GOOD, COLOR_ERROR,
)
from ..core.geometry import dist2d, box_center, angle_between

COLOR_WHITE = (255, 255, 255)

import cv2

_DEFAULTS = {
    "knee_bend_threshold_deg": 160,   # below this while airborne = bent
    "min_airborne_height_cm": 10,     # must be at least 10cm off ground to count as airborne
}


class JumpAnalyzer(BaseDrillAnalyzer):
    """Analyzer for vertical / countermovement jump drills."""

    drill_name = "Jump Test"

    def __init__(self, config: DrillConfig):
        super().__init__(config)
        p = {**_DEFAULTS, **config.drill_params}
        self.knee_bend_threshold: float = p["knee_bend_threshold_deg"]
        self.min_airborne_cm: float = p["min_airborne_height_cm"]

    def setup(self) -> None:
        # ── Time ──
        self.start_frame: Optional[int] = None
        self.end_frame: Optional[int] = None

        # ── Ankle height tracking (for jump measurement) ──
        self.start_ankle_y: Optional[float] = None  # ground reference
        self.current_jump_cm: float = 0.0
        self.max_jump_cm: float = 0.0

        # ── Knee angles ──
        self.all_left_angles: list[float] = []
        self.all_right_angles: list[float] = []
        self.current_l_angle: Optional[float] = None
        self.current_r_angle: Optional[float] = None

        # ── Knee bend events (during jump) ──
        self.knee_bent_during_jump: bool = False
        self.knee_bend_events: list[dict] = []

    # ── hooks ──

    def on_object_detected(self, frame_num, class_name, box, stable_id, confidence):
        pass  # Jump test doesn't track objects (ball is only used for calibration)

    def on_pose_estimated(self, frame_num, keypoints, player_box, track_id):
        # --- Main Player Tracking Filter ---
        if not hasattr(self, 'main_player_id'):
            self.main_player_id = None
        
        box_area = (player_box[2] - player_box[0]) * (player_box[3] - player_box[1])
        img_area = self.frame_width * self.frame_height if hasattr(self, 'frame_width') else 1920*1080

        # Ignore tiny people in background (less than 2% of screen)
        if img_area > 0 and (box_area / img_area) < 0.02:
            return

        if self.main_player_id is None:
            self.main_player_id = track_id
        elif track_id != self.main_player_id:
            return

        # -----------------------------------

        if self.start_frame is None:
            self.start_frame = frame_num
        self.end_frame = frame_num

        conf = self.config.keypoint_confidence
        cal = self.calibration

        # ── Get ankle positions ──
        l_ankle = keypoints[KP_L_ANKLE]
        r_ankle = keypoints[KP_R_ANKLE]

        if l_ankle[2] < conf or r_ankle[2] < conf:
            return  # Need both ankles

        avg_ankle_y = (float(l_ankle[1]) + float(r_ankle[1])) / 2.0

        # Update ground reference dynamically to the lowest point (highest Y value)
        # This fixes negative jump values if the player walks towards the camera before jumping
        if self.start_ankle_y is None or avg_ankle_y > self.start_ankle_y:
            self.start_ankle_y = avg_ankle_y

        # Current jump height (pixels → cm via calibration)
        jump_px = self.start_ankle_y - avg_ankle_y  # positive = up
        if cal.is_calibrated and cal.pixels_per_meter:
            self.current_jump_cm = (jump_px / cal.pixels_per_meter) * 100  # m → cm
        else:
            self.current_jump_cm = jump_px  # fallback: pixels

        if self.current_jump_cm > self.max_jump_cm:
            self.max_jump_cm = self.current_jump_cm

        # ── Knee angles ──
        l_hip = keypoints[KP_L_HIP]
        l_knee = keypoints[KP_L_KNEE]
        r_hip = keypoints[KP_R_HIP]
        r_knee = keypoints[KP_R_KNEE]

        self.current_l_angle = None
        self.current_r_angle = None

        if all(kp[2] > conf for kp in [l_hip, l_knee, l_ankle]):
            self.current_l_angle = angle_between(
                (l_hip[0], l_hip[1]), (l_knee[0], l_knee[1]), (l_ankle[0], l_ankle[1])
            )
            self.all_left_angles.append(self.current_l_angle)

        if all(kp[2] > conf for kp in [r_hip, r_knee, r_ankle]):
            self.current_r_angle = angle_between(
                (r_hip[0], r_hip[1]), (r_knee[0], r_knee[1]), (r_ankle[0], r_ankle[1])
            )
            self.all_right_angles.append(self.current_r_angle)

        # ── Knee bend detection (ONLY when airborne > 10cm) ──
        if self.current_jump_cm > self.min_airborne_cm:
            bent_this_frame = False
            if self.current_l_angle is not None and self.current_l_angle < self.knee_bend_threshold:
                bent_this_frame = True
            if self.current_r_angle is not None and self.current_r_angle < self.knee_bend_threshold:
                bent_this_frame = True

            if bent_this_frame:
                self.knee_bent_during_jump = True
                self.knee_bend_events.append({
                    "frame": frame_num,
                    "left_knee_angle": round(self.current_l_angle, 1) if self.current_l_angle else None,
                    "right_knee_angle": round(self.current_r_angle, 1) if self.current_r_angle else None,
                    "jump_height_cm": round(self.current_jump_cm, 2),
                })

    def compute_drill_metrics(self, frame_num):
        # Abort early if the ball or player is not detected for calibration
        check_frame = min(30, self.total_frames) if self.total_frames > 0 else 30
        if frame_num == check_frame:
            if self.calibration.ball.sample_count == 0:
                raise ValueError("لا يمكن رؤية الكرة للمعايرة. يرجى التأكد من وجود كرة بجانب اللاعب.")
            if self.calibration.player.sample_count == 0:
                raise ValueError("لا يمكن رؤية اللاعب بوضوح. يرجى التأكد من ظهور اللاعب كاملًا.")


    def build_overlay(self, frame_num):
        self.panel.clear()

        # Jump height
        color = COLOR_GOOD if self.current_jump_cm > 0 else COLOR_WHITE
        self.panel.add(f"Jump: {self.current_jump_cm:.1f} cm", color)

        # L Knee
        if self.current_l_angle is not None:
            color = COLOR_ERROR if self.current_l_angle < self.knee_bend_threshold else COLOR_GOOD
            self.panel.add(f"L Knee: {self.current_l_angle:.0f}\u00b0", color)

        # R Knee
        if self.current_r_angle is not None:
            color = COLOR_ERROR if self.current_r_angle < self.knee_bend_threshold else COLOR_GOOD
            self.panel.add(f"R Knee: {self.current_r_angle:.0f}\u00b0", color)

        # KNEE BENT alert
        if self.current_jump_cm > self.min_airborne_cm:
            bent = False
            if self.current_l_angle and self.current_l_angle < self.knee_bend_threshold:
                bent = True
            if self.current_r_angle and self.current_r_angle < self.knee_bend_threshold:
                bent = True
            if bent:
                self.panel.add("KNEE BENT!", COLOR_ERROR)

    def draw_custom(self, frame, frame_num):
        # Draw big "KNEE BENT!" text in center when detected
        if self.current_jump_cm > self.min_airborne_cm:
            bent = False
            if self.current_l_angle and self.current_l_angle < self.knee_bend_threshold:
                bent = True
            if self.current_r_angle and self.current_r_angle < self.knee_bend_threshold:
                bent = True
            if bent:
                h, w = frame.shape[:2]
                cv2.putText(frame, "KNEE BENT!", (w // 2 - 100, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        return frame

    def generate_report(self) -> dict:
        self.calibration.compute()
        is_cal = self.calibration.is_calibrated

        duration = 0
        if self.start_frame and self.end_frame:
            duration = (self.end_frame - self.start_frame) / self.fps

        # Knee angle stats
        l_knee = {}
        if self.all_left_angles:
            l_knee = {
                "average_angle": round(float(np.mean(self.all_left_angles)), 1),
                "min_angle": round(float(np.min(self.all_left_angles)), 1),
                "max_angle": round(float(np.max(self.all_left_angles)), 1),
            }

        r_knee = {}
        if self.all_right_angles:
            r_knee = {
                "average_angle": round(float(np.mean(self.all_right_angles)), 1),
                "min_angle": round(float(np.min(self.all_right_angles)), 1),
                "max_angle": round(float(np.max(self.all_right_angles)), 1),
            }

        # Form assessment
        if not self.knee_bent_during_jump:
            form = "Good Form: Legs kept straight"
        else:
            form = f"Warning: Player bent knees during jump ({len(self.knee_bend_events)} frames)"

        return {
            "drill_info": {},
            "time_metrics": {"total_duration_seconds": round(duration, 2)},
            "jump_metrics": {
                "max_jump_height_cm": round(self.max_jump_cm, 2),
            },
            "knee_angle_metrics": {
                "left_knee": l_knee,
                "right_knee": r_knee,
                "knee_bent_during_jump": self.knee_bent_during_jump,
                "knee_bend_events_count": len(self.knee_bend_events),
                "knee_bend_events": self.knee_bend_events,
            },
            "form_assessment": form,
        }

