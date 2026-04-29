"""
weakfoot.py — Weak Foot Dribble & Shoot Drill Analyzer.

Faithfully ports all logic from weakfoot_ultimate.py into the
BaseDrillAnalyzer architecture.

Measurements:
  • Ball distance (ankle-to-ball, meters)
  • Ball control classification (Good / Loose / Far)
  • Touch counting (far→close edge trigger)
  • Active foot per frame (L/R usage %)
  • Player speed (m/s) with smoothing
  • Ball speed (m/s)
  • Shot kinematics (knee angle, angular velocity, trunk rotation)
  • Balance score (shoulder-hip lateral sway)
  • Phase detection (dribbling / shooting / idle)
  • Overall scoring (control% + shot_power + shot_technique + balance)
"""

from __future__ import annotations

import math
import numpy as np
from typing import Optional
from collections import deque

from ..base_drill import BaseDrillAnalyzer
from ..config import DrillConfig
from ..constants import (
    KP_NOSE,
    KP_L_ANKLE, KP_R_ANKLE, KP_L_KNEE, KP_R_KNEE,
    KP_L_HIP, KP_R_HIP, KP_L_SHOULDER, KP_R_SHOULDER,
    COLOR_GOOD, COLOR_ERROR,
)
from ..core.geometry import dist2d, box_center, angle_between
from ..core.ball_physics import BallVelocityTracker


# ── Defaults (overridable via config.drill_params or YAML) ──
_DEFAULTS = {
    "weak_foot": "left",                 # which foot is the weak foot
    "good_control_threshold_m": 0.5,
    "loose_ball_threshold_m": 1.0,
    "weak_shot_threshold_m": 1.3,
    "smooth_window": 10,
    "min_movement_px": 3.0,
    "kp_conf": 0.3,                      # keypoint confidence
    "velocity_spike_threshold": 3.0,
    "direction_change_threshold_deg": 30.0,
}

# Colors for overlay
COLOR_LOOSE = (0, 165, 255)   # orange
COLOR_FAR   = (0, 0, 255)     # red
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (255, 255, 0)
COLOR_CYAN  = (0, 255, 255)
COLOR_ACCENT = (255, 200, 100)


class WeakFootAnalyzer(BaseDrillAnalyzer):
    """Analyzer for the weak foot dribble + shot drill.

    Faithfully matches weakfoot_ultimate.py logic.
    """

    drill_name = "Weak Foot Drill"

    def __init__(self, config: DrillConfig):
        super().__init__(config)
        p = {**_DEFAULTS, **config.drill_params}
        self.weak_foot: str          = p["weak_foot"]
        self.good_control_m: float   = p["good_control_threshold_m"]
        self.loose_ball_m: float     = p["loose_ball_threshold_m"]
        self.weak_shot_m: float      = p["weak_shot_threshold_m"]
        self.smooth_win: int         = p["smooth_window"]
        self.min_movement_px: float  = p["min_movement_px"]
        self.CONF: float             = p["kp_conf"]
        self._vel_spike_thr: float   = p["velocity_spike_threshold"]
        self._dir_change_thr: float  = p["direction_change_threshold_deg"]

    def setup(self) -> None:
        # ── Player position & speed ──
        self.smoothed_pos: Optional[tuple] = None
        self.player_positions: list[tuple] = []
        self.player_speeds: list[float] = []

        # ── Ball ──
        self.ball_positions: list[tuple] = []
        self.ball_frame_idxs: list[int] = []
        self.ball_player_distances: list[float] = []  # pixel distances
        self.ball_visible_frames: int = 0
        self.max_ball_speed_px: float = 0

        # ── Cones ──
        self.cone_positions: dict[int, tuple] = {}
        self.cone_proximities: list[float] = []

        # ── Dribble control ──
        self.good_control_frames: int = 0
        self.loose_ball_frames: int = 0
        self.far_ball_frames: int = 0

        # ── Active foot (per-frame) ──
        self.left_foot_active_frames: int = 0
        self.right_foot_active_frames: int = 0

        # ── Touch counter (edge-triggered) ──
        self._was_close: bool = False
        self.touch_count: int = 0
        self.ball_velocity_tracker = BallVelocityTracker()

        # ── Shot kinematics ──
        self.knee_angles: list[float] = []
        self.trunk_angles: list[float] = []
        self.knee_ang_vel: list[float] = []
        self.shot_detected: bool = False
        self.shot_frame: Optional[int] = None

        # ── Shooting zone ──
        self.shooting_zone_active: bool = False

        # ── Phase ──
        self.current_phase: str = "idle"
        self.dribble_speeds: list[float] = []

        # ── Balance (trunk sway) ──
        self.trunk_sway_values: list[float] = []

        # ── Per-frame live state ──
        self._ball_pos: Optional[tuple] = None
        self._ball_box = None
        self._active_foot: Optional[str] = None
        self._foot_pt: Optional[tuple] = None
        self._control_info = None  # (dist_m, status, color)

    # ── hooks ──

    def on_object_detected(self, frame_num, class_name, box, stable_id, confidence):
        if class_name.lower() in self.config.ball_class_names:
            self._ball_box = box
            self._ball_pos = box_center(box)
        else:
            self.cone_positions[stable_id] = box_center(box)

    def on_pose_estimated(self, frame_num, keypoints, player_box, track_id):
        raw = box_center(player_box)
        alpha = self.config.position_smoothing

        if self.smoothed_pos is None:
            self.smoothed_pos = raw
        else:
            self.smoothed_pos = (
                alpha * raw[0] + (1 - alpha) * self.smoothed_pos[0],
                alpha * raw[1] + (1 - alpha) * self.smoothed_pos[1],
            )
        self.player_positions.append(self.smoothed_pos)

        # Player speed
        if len(self.player_positions) >= 2:
            d = dist2d(self.player_positions[-2], self.player_positions[-1])
            speed = d if d >= self.min_movement_px else 0
            self.player_speeds.append(speed)
            if self.current_phase == "dribbling":
                self.dribble_speeds.append(speed)

        # Cone proximity
        if self.cone_positions:
            self.cone_proximities.append(
                min(dist2d(self.smoothed_pos, p) for p in self.cone_positions.values())
            )

        # Shooting zone gate
        if not self.shooting_zone_active and self.cone_positions:
            cone_xs = [p[0] for p in self.cone_positions.values()]
            if len(cone_xs) >= 2:
                if self.smoothed_pos[0] < min(cone_xs):
                    self.shooting_zone_active = True

    def compute_drill_metrics(self, frame_num):
        kp = self.current_keypoints
        ppm = self.calibration.pixels_per_meter

        # ── Ball position update ──
        if self._ball_box is not None:
            bc = self._ball_pos
            self.ball_positions.append(bc)
            self.ball_frame_idxs.append(frame_num)
            self.ball_visible_frames += 1
            self.ball_velocity_tracker.update(frame_num, bc)

            # Distance: ankle-to-ball (closest ankle), fallback to player center
            min_dist = None
            if kp is not None and len(kp) > 16:
                dists = []
                la, ra = kp[KP_L_ANKLE], kp[KP_R_ANKLE]
                if la[2] > self.CONF:
                    dists.append(dist2d((la[0], la[1]), bc))
                if ra[2] > self.CONF:
                    dists.append(dist2d((ra[0], ra[1]), bc))
                if dists:
                    min_dist = min(dists)
            if min_dist is None and self.smoothed_pos is not None:
                min_dist = dist2d(self.smoothed_pos, bc)
            if min_dist is not None:
                self.ball_player_distances.append(min_dist)

            # Ball speed (consecutive frames only)
            if len(self.ball_positions) >= 2 and len(self.ball_frame_idxs) >= 2:
                if self.ball_frame_idxs[-1] - self.ball_frame_idxs[-2] == 1:
                    spd = dist2d(self.ball_positions[-2], self.ball_positions[-1])
                    if spd > self.max_ball_speed_px:
                        self.max_ball_speed_px = spd

        # ── Active foot + dribble control ──
        self._active_foot = None
        self._foot_pt = None
        self._control_info = None

        if kp is not None and self._ball_pos is not None and len(kp) >= 17:
            la, ra = kp[KP_L_ANKLE], kp[KP_R_ANKLE]
            l_dist = dist2d((la[0], la[1]), self._ball_pos) if la[2] > self.CONF else float("inf")
            r_dist = dist2d((ra[0], ra[1]), self._ball_pos) if ra[2] > self.CONF else float("inf")
            closest_dist = min(l_dist, r_dist)

            # Active foot (match old logic exactly)
            if l_dist < r_dist and la[2] > self.CONF:
                self._active_foot = "left"
                self._foot_pt = (int(la[0]), int(la[1]))
                self.left_foot_active_frames += 1
            elif ra[2] > self.CONF:
                self._active_foot = "right"
                self._foot_pt = (int(ra[0]), int(ra[1]))
                self.right_foot_active_frames += 1
            elif la[2] > self.CONF:
                self._active_foot = "left"
                self._foot_pt = (int(la[0]), int(la[1]))
                self.left_foot_active_frames += 1

            if self._active_foot is not None:
                dist_m = self.calibration.px_to_m(closest_dist)

                # Control classification
                if dist_m < self.good_control_m:
                    self.good_control_frames += 1
                    status, status_color = "Good Control", COLOR_GOOD
                elif dist_m < self.loose_ball_m:
                    self.loose_ball_frames += 1
                    status, status_color = "Loose Ball", COLOR_LOOSE
                else:
                    self.far_ball_frames += 1
                    status, status_color = "Far", COLOR_FAR

                self._control_info = (dist_m, status, status_color)

                # Touch counter: far→close = 1 touch + velocity check
                is_close = dist_m < self.loose_ball_m
                physics_touch = self.ball_velocity_tracker.has_physical_touch(
                    velocity_threshold=self._vel_spike_thr,
                    direction_threshold_deg=self._dir_change_thr,
                )
                if not self._was_close and is_close and physics_touch:
                    self.touch_count += 1
                self._was_close = is_close

        # ── Shot kinematics ──
        if kp is not None and len(kp) >= 17:
            self._compute_balance(kp)
            if self.shooting_zone_active:
                self._compute_shot_kinematics(kp)

        # ── Phase detection ──
        self._detect_phase(frame_num)

        # ── Reset per-frame ball state ──
        self._ball_box = None

    def build_overlay(self, frame_num):
        self.panel.clear()
        S = self.smooth_win
        cal = self.calibration

        # Ball distance
        if self.ball_player_distances:
            recent = self.ball_player_distances[-S:]
            avg_d = float(np.mean(recent))
            if cal.is_calibrated:
                avg_m = cal.px_to_m(avg_d)
                stat = "Good" if avg_m < 0.5 else ("Loose" if avg_m < 1.0 else "Far!")
                c = COLOR_GOOD if avg_m < 0.5 else (COLOR_LOOSE if avg_m < 1.0 else COLOR_FAR)
                self.panel.add(f"Ball: {avg_m:.2f}m ({stat})", c)
            else:
                self.panel.add(f"Ball: {avg_d:.0f}px", COLOR_GOOD)

        # Player speed
        if self.player_speeds:
            recent_spd = self.player_speeds[-S:]
            avg_spd = float(np.mean(recent_spd))
            if cal.is_calibrated:
                spd_ms = cal.px_to_m(avg_spd * self.fps)
                self.panel.add(f"Speed: {spd_ms:.1f} m/s", COLOR_WHITE)
            else:
                self.panel.add(f"Speed: {avg_spd * self.fps:.0f} px/s", COLOR_WHITE)

        # Ball speed (only consecutive frames — avoids spike when ball reappears)
        if (len(self.ball_frame_idxs) >= 2
                and self.ball_frame_idxs[-1] - self.ball_frame_idxs[-2] == 1):
            bspd_px = dist2d(self.ball_positions[-2], self.ball_positions[-1])
            if cal.is_calibrated:
                bspd_m = cal.px_to_m(bspd_px) * self.fps
                if bspd_m < 45:  # sanity cap: world record shot ≈ 40 m/s
                    self.panel.add(f"Ball Spd: {bspd_m:.1f} m/s", COLOR_WHITE)

        # Balance
        if self.trunk_sway_values and cal.is_calibrated:
            recent_sway = self.trunk_sway_values[-S:]
            avg_sway = float(np.mean(recent_sway))
            sway_m = cal.px_to_m(avg_sway)
            bal_score = max(0, min(100, 100 - (sway_m / 0.3 * 100)))
            bal_c = COLOR_GOOD if bal_score > 70 else (COLOR_LOOSE if bal_score > 40 else COLOR_FAR)
            self.panel.add(f"Balance: {bal_score:.0f}", bal_c)

        # Phase
        phase_colors = {"dribbling": COLOR_GOOD, "shooting": COLOR_CYAN, "idle": (128, 128, 128)}
        self.panel.add(
            f"Phase: {self.current_phase.upper()}",
            phase_colors.get(self.current_phase, COLOR_WHITE),
        )

        # Touch count
        self.panel.add(f"Touches: {self.touch_count}", COLOR_YELLOW)

        # Shot alert
        if self.shot_detected:
            self.panel.add("SHOT DETECTED!", COLOR_CYAN)

    def draw_custom(self, frame, frame_num):
        import cv2
        from ..visualization.drawing import draw_ball_foot_line

        # Foot-to-ball line with distance label
        if self._foot_pt is not None and self._ball_pos is not None and self._control_info is not None:
            dist_m, status, line_color = self._control_info
            ball_int = (int(self._ball_pos[0]), int(self._ball_pos[1]))
            cv2.line(frame, self._foot_pt, ball_int, line_color, 2, cv2.LINE_AA)

            mid = ((self._foot_pt[0] + ball_int[0]) // 2,
                   (self._foot_pt[1] + ball_int[1]) // 2)
            if self.calibration.is_calibrated:
                cv2.putText(frame, f"{dist_m:.2f}m", mid,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 2, cv2.LINE_AA)

            # Foot label
            is_weak = (self._active_foot == self.weak_foot)
            foot_label = f"{self._active_foot[0].upper()} (weak)" if is_weak else self._active_foot[0].upper()
            cv2.putText(frame, f"Foot: {foot_label}",
                        (self._foot_pt[0] - 30, self._foot_pt[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_YELLOW, 1, cv2.LINE_AA)

        return frame

    def generate_report(self) -> dict:
        self.calibration.compute()
        cal = self.calibration.is_calibrated
        ppm = self.calibration.pixels_per_meter

        total_ctrl = self.good_control_frames + self.loose_ball_frames + self.far_ball_frames
        ctrl_pct = (self.good_control_frames / total_ctrl * 100) if total_ctrl > 0 else 0
        possession_pct = (self.ball_visible_frames / max(self.total_frames, 1) * 100)

        avg_ball_dist = float(np.mean(self.ball_player_distances)) if self.ball_player_distances else 0
        avg_ball_dist_m = self.calibration.px_to_m(avg_ball_dist) if cal else avg_ball_dist

        # Speed
        speed_metrics = {}
        if self.player_speeds:
            avg_ppf = float(np.mean(self.player_speeds))
            max_ppf = float(np.max(self.player_speeds))
            dribble_avg = float(np.mean(self.dribble_speeds)) if self.dribble_speeds else 0
            if cal:
                speed_metrics = {
                    "average_speed_m_per_s": round(self.calibration.px_to_m(avg_ppf * self.fps), 2),
                    "max_speed_m_per_s": round(self.calibration.px_to_m(max_ppf * self.fps), 2),
                    "dribble_phase_avg_speed_m_per_s": round(self.calibration.px_to_m(dribble_avg * self.fps), 2),
                }

        # Shot metrics
        shot_metrics = {}
        if self.knee_ang_vel:
            max_kv = max(self.knee_ang_vel)
            avg_tr = float(np.mean(self.trunk_angles)) if self.trunk_angles else 0
            backswing = (max(self.knee_angles) - min(self.knee_angles)) if self.knee_angles else 0
            shot_power_score = min(100, max_kv * 5)
            shot_tech_score = min(100, backswing + avg_tr)
            max_ball_speed_m = self.calibration.px_to_m(self.max_ball_speed_px) * self.fps if cal else self.max_ball_speed_px * self.fps

            feedback = []
            if avg_tr < 15:
                feedback.append("Low trunk rotation — rotate hips/shoulders more into the shot")
            if backswing < 30:
                feedback.append("Short backswing — pull kicking leg further back")
            if max_kv < 5:
                feedback.append("Weak swing — accelerate leg through the ball faster")
            if self.far_ball_frames > 5:
                feedback.append("Over-stretching — approach the ball closer before shooting")

            shot_metrics = {
                "shot_detected": self.shot_detected,
                "shot_frame": int(self.shot_frame) if self.shot_frame else None,
                "max_knee_angular_velocity": round(float(max_kv), 2),
                "average_trunk_rotation": round(float(avg_tr), 2),
                "backswing_angle": round(float(backswing), 2),
                "shot_power_score": round(float(shot_power_score), 1),
                "shot_technique_score": round(float(shot_tech_score), 1),
                "max_ball_speed_m_per_s": round(float(max_ball_speed_m), 2),
                "feedback": feedback if feedback else ["Good form — no major issues detected"],
            }
        else:
            shot_metrics = {
                "shot_detected": False,
                "feedback": ["Insufficient data for shot analysis"],
            }

        # Balance
        balance_metrics = {}
        if self.trunk_sway_values:
            avg_sway_px = float(np.mean(self.trunk_sway_values))
            std_sway_px = float(np.std(self.trunk_sway_values))
            avg_sway_m = self.calibration.px_to_m(avg_sway_px) if cal else avg_sway_px
            balance_score = max(0, min(100, 100 - (avg_sway_m / 0.3 * 100))) if cal else 50
            balance_metrics = {
                "average_sway_px": round(avg_sway_px, 2),
                "sway_std_px": round(std_sway_px, 2),
                "balance_score": round(float(balance_score), 1),
            }

        # Foot usage
        total_foot = self.left_foot_active_frames + self.right_foot_active_frames
        foot_usage = {}
        if total_foot > 0:
            weak_frames = self.left_foot_active_frames if self.weak_foot == "left" else self.right_foot_active_frames
            foot_usage = {
                "left_percentage": round(self.left_foot_active_frames / total_foot * 100, 1),
                "right_percentage": round(self.right_foot_active_frames / total_foot * 100, 1),
                "weak_foot": self.weak_foot,
                "weak_foot_usage_percentage": round(weak_frames / total_foot * 100, 1),
            }

        # Cone metrics
        cone_metrics = {}
        if self.cone_proximities and cal:
            mind = float(np.min(self.cone_proximities))
            cone_metrics = {"closest_approach_meters": round(self.calibration.px_to_m(mind), 2)}

        # Overall score (match old formula)
        scores = []
        if total_ctrl > 0:
            scores.append(ctrl_pct)
        if "shot_power_score" in shot_metrics:
            scores.append(shot_metrics["shot_power_score"])
            scores.append(shot_metrics["shot_technique_score"])
        if "balance_score" in balance_metrics:
            scores.append(balance_metrics["balance_score"])
        overall = float(np.mean(scores)) if scores else 0

        # Form assessment
        if overall >= 75:
            assessment = "Strong weak foot performance"
        elif overall >= 50:
            assessment = "Acceptable weak foot — room for improvement"
        else:
            assessment = "Weak foot needs significant work"

        return {
            "drill_info": {"weak_foot": self.weak_foot},
            "dribbling_metrics": {
                "good_control_frames": self.good_control_frames,
                "loose_ball_frames": self.loose_ball_frames,
                "far_ball_frames": self.far_ball_frames,
                "control_percentage": round(ctrl_pct, 1),
                "ball_possession_percentage": round(possession_pct, 1),
                "average_ball_distance_m": round(float(avg_ball_dist_m), 3),
                "touch_count": self.touch_count,
                "unit": "meters" if cal else "pixels",
            },
            "speed_metrics": speed_metrics,
            "shot_metrics": shot_metrics,
            "balance_metrics": balance_metrics,
            "foot_usage": foot_usage,
            "cone_metrics": cone_metrics,
            "overall_score": round(overall, 1),
            "form_assessment": assessment,
        }

    # ── internal ──

    def _compute_balance(self, kp):
        """Trunk sway — shoulder center vs hip center lateral offset."""
        lsh, rsh = kp[KP_L_SHOULDER], kp[KP_R_SHOULDER]
        lhip, rhip = kp[KP_L_HIP], kp[KP_R_HIP]
        if all(kp_pt[2] > self.CONF for kp_pt in [lsh, rsh, lhip, rhip]):
            shc = ((lsh[0] + rsh[0]) / 2, (lsh[1] + rsh[1]) / 2)
            hipc = ((lhip[0] + rhip[0]) / 2, (lhip[1] + rhip[1]) / 2)
            lateral_offset = abs(shc[0] - hipc[0])
            self.trunk_sway_values.append(lateral_offset)

    def _compute_shot_kinematics(self, kp):
        """Only runs when shooting_zone_active is True (player passed cones)."""
        # Select weak foot joints
        if self.weak_foot == "left":
            hip, knee, ankle = kp[KP_L_HIP], kp[KP_L_KNEE], kp[KP_L_ANKLE]
        else:
            hip, knee, ankle = kp[KP_R_HIP], kp[KP_R_KNEE], kp[KP_R_ANKLE]

        # Knee angle
        if hip[2] > self.CONF and knee[2] > self.CONF and ankle[2] > self.CONF:
            ka = angle_between(
                (hip[0], hip[1]), (knee[0], knee[1]), (ankle[0], ankle[1])
            )
            self.knee_angles.append(ka)
            if len(self.knee_angles) > 1:
                self.knee_ang_vel.append(abs(self.knee_angles[-1] - self.knee_angles[-2]))

        # Trunk rotation
        lsh, rsh = kp[KP_L_SHOULDER], kp[KP_R_SHOULDER]
        lhip, rhip = kp[KP_L_HIP], kp[KP_R_HIP]
        if all(kp_pt[2] > self.CONF for kp_pt in [lsh, rsh, lhip, rhip]):
            shc = ((lsh[0] + rsh[0]) / 2, (lsh[1] + rsh[1]) / 2)
            hipc = ((lhip[0] + rhip[0]) / 2, (lhip[1] + rhip[1]) / 2)
            vert = (hipc[0], hipc[1] - 100)
            ta = angle_between(vert, hipc, shc)
            self.trunk_angles.append(ta)

        # Shot detection heuristic
        if (len(self.ball_player_distances) > 5
                and len(self.knee_ang_vel) > 5
                and not self.shot_detected):
            recent_dist = np.mean(self.ball_player_distances[-5:])
            if (self.ball_player_distances[-1] < 0.7 * recent_dist
                    and max(self.knee_ang_vel[-5:]) > 5):
                self.shot_detected = True
                self.shot_frame = len(self.knee_ang_vel)

    def _detect_phase(self, frame_idx):
        """Classify current phase as dribbling / shooting / idle."""
        if not self.ball_player_distances:
            self.current_phase = "idle"
        elif (self.shot_detected and self.shot_frame
              and abs(len(self.knee_ang_vel) - self.shot_frame) < 15):
            self.current_phase = "shooting"
        elif self.ball_player_distances[-1] < (
            self.calibration.pixels_per_meter * self.loose_ball_m
            if self.calibration.is_calibrated else 150
        ):
            self.current_phase = "dribbling"
        else:
            self.current_phase = "idle"
