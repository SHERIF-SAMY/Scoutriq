"""
jumping_15.py — Jumping 15 Drill Analyzer.

Analyzes side-to-side jumps over a ball for a fixed duration (default 15s).
Key Metrics:
 - Jump Count: X-coordinate crossings of the ball.
 - Knee Symmetry: Both knees must bend equally in the air.
 - Pace Consistency: Time taken per jump (like Diamond drill pace).
"""

from __future__ import annotations

import math
import numpy as np
import cv2
from typing import Optional

from ..base_drill import BaseDrillAnalyzer
from ..config import DrillConfig
from ..constants import (
    KP_L_KNEE, KP_R_KNEE, KP_L_HIP, KP_R_HIP, KP_L_ANKLE, KP_R_ANKLE,
    COLOR_GOOD, COLOR_ERROR
)
from ..core.geometry import box_center, angle_between

COLOR_WHITE = (255, 255, 255)
COLOR_WARNING = (0, 165, 255)  # Orange

_DEFAULTS = {
    "drill_duration_s": 15.0,
    "asymmetry_threshold_deg": 40.0,  # Max allowed angle diff between knees in air
    "crossing_buffer_px": 5.0,        # Prevention of jitter-based multi-counts
}


class Jumping15Analyzer(BaseDrillAnalyzer):
    """Analyzer for side-to-side jumping drill."""

    drill_name = "Jumping 15"

    def __init__(self, config: DrillConfig):
        super().__init__(config)
        p = {**_DEFAULTS, **config.drill_params}
        self.target_duration: float = p["drill_duration_s"]
        self.asym_thresh: float = p["asymmetry_threshold_deg"]
        self.crossing_buffer: float = p["crossing_buffer_px"]

    def setup(self) -> None:
        # ── Time / Duration ──
        self.start_frame: Optional[int] = None
        self.end_frame: Optional[int] = None
        self.time_up: bool = False

        # ── Ball tracking ──
        self.ball_x: Optional[float] = None  # Reference point for crossing

        # ── Jump counting ──
        self.player_x_history: list[float] = []
        self.current_side: Optional[str] = None  # "left" or "right"
        self.jump_count: int = 0
        self.jump_times: list[dict] = []  # {jump_num, frame, duration_s}
        self.last_jump_frame: Optional[int] = None

        # ── Knee symmetry tracking ──
        self.current_l_angle: Optional[float] = None
        self.current_r_angle: Optional[float] = None
        self.asymmetry_events: list[dict] = []
        self.is_asymmetric_now: bool = False

    # ── hooks ──

    def on_object_detected(self, frame_num, class_name, box, stable_id, confidence):
        # We only care about the ball's X position for jump crossing reference
        if class_name.lower() in self.config.ball_class_names:
            area = (box[2] - box[0]) * (box[3] - box[1])
            
            # If new frame, apply previous frame's best ball to self.ball_x
            if frame_num != getattr(self, '_last_ball_frame', -1):
                if getattr(self, '_biggest_ball_x', None) is not None:
                    if self.ball_x is None:
                        self.ball_x = self._biggest_ball_x
                    else:
                        alpha = 0.3  # Smooth but responsive to camera pans
                        self.ball_x = alpha * self._biggest_ball_x + (1 - alpha) * self.ball_x
                
                self._last_ball_frame = frame_num
                self._biggest_ball_area = 0
                self._biggest_ball_x = None
                
            # Track largest ball this frame to avoid false positives (like socks)
            if area > getattr(self, '_biggest_ball_area', 0):
                self._biggest_ball_area = area
                self._biggest_ball_x = box_center(box)[0]
                
                # Provisionally set for the very first frame
                if self.ball_x is None:
                    self.ball_x = self._biggest_ball_x

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
            self.last_jump_frame = frame_num

        self.end_frame = frame_num

        # (Removed 15s fixed duration check at user request. Analyzes whole video)

        conf = self.config.keypoint_confidence

        # ── Player X Position (use average of hips) ──
        l_hip = keypoints[KP_L_HIP]
        r_hip = keypoints[KP_R_HIP]
        
        if l_hip[2] > conf and r_hip[2] > conf:
            player_x = (l_hip[0] + r_hip[0]) / 2.0
            self.player_x_history.append(player_x)

            # Detect crossing (jump)
            if self.ball_x is not None:
                # Add buffer to prevent jitter noise when standing right over the ball
                if player_x < (self.ball_x - self.crossing_buffer):
                    new_side = "left"
                elif player_x > (self.ball_x + self.crossing_buffer):
                    new_side = "right"
                else:
                    new_side = self.current_side  # In buffer zone, keep old side

                if self.current_side is not None and new_side != self.current_side:
                    # Traversed across the ball -> Counted a jump
                    self.jump_count += 1
                    t_jump = (frame_num - self.last_jump_frame) / self.fps
                    self.jump_times.append({
                        "jump_num": self.jump_count,
                        "frame": frame_num,
                        "duration_s": round(t_jump, 3)
                    })
                    self.last_jump_frame = frame_num

                self.current_side = new_side

        # ── Knee Angles (Symmetry check) ──
        l_knee = keypoints[KP_L_KNEE]
        r_knee = keypoints[KP_R_KNEE]
        l_ankle = keypoints[KP_L_ANKLE]
        r_ankle = keypoints[KP_R_ANKLE]

        self.current_l_angle = None
        self.current_r_angle = None
        self.is_asymmetric_now = False

        if all(kp[2] > conf for kp in [l_hip, l_knee, l_ankle]):
            self.current_l_angle = angle_between(
                (l_hip[0], l_hip[1]), (l_knee[0], l_knee[1]), (l_ankle[0], l_ankle[1])
            )

        if all(kp[2] > conf for kp in [r_hip, r_knee, r_ankle]):
            self.current_r_angle = angle_between(
                (r_hip[0], r_hip[1]), (r_knee[0], r_knee[1]), (r_ankle[0], r_ankle[1])
            )

        # Only check symmetry if we have both angles and the drill has started (jump_count > 0)
        if self.current_l_angle is not None and self.current_r_angle is not None and self.jump_count > 0:
            diff = abs(self.current_l_angle - self.current_r_angle)
            if diff > self.asym_thresh:
                self.is_asymmetric_now = True
                self.asymmetry_events.append({
                    "frame": frame_num,
                    "jump_num": self.jump_count,
                    "left_knee_deg": round(self.current_l_angle, 1),
                    "right_knee_deg": round(self.current_r_angle, 1),
                    "diff_deg": round(diff, 1)
                })

    def compute_drill_metrics(self, frame_num):
        # Abort early if the ball or player is not detected, since ball is needed for crossing reference
        check_frame = min(30, self.total_frames) if hasattr(self, 'total_frames') and self.total_frames > 0 else 30
        if frame_num == check_frame:
            if self.ball_x is None:
                raise ValueError("لا يمكن رؤية الكرة. يرجى التأكد من وجود كرة كمؤشر للقفز الجانبي.")
            if getattr(self, 'main_player_id', None) is None:
                raise ValueError("لا يمكن رؤية اللاعب بوضوح. يرجى التأكد من ظهور اللاعب كاملًا.")

    def build_overlay(self, frame_num):
        self.panel.clear()

        # Timer
        elapsed = 0.0
        if self.start_frame:
            elapsed = (frame_num - self.start_frame) / self.fps
            self.panel.add(f"Time: {elapsed:.1f}s", COLOR_WHITE)

        # Jump Count
        self.panel.add(f"Jumps: {self.jump_count}", COLOR_GOOD)

        # Side
        if self.current_side:
            self.panel.add(f"Side: {self.current_side.upper()}", COLOR_WHITE)

        # Knee Symmetry Alert
        if self.is_asymmetric_now:
            self.panel.add(f"ASYMMETRIC KNEES!", COLOR_ERROR)

        # Recent Pace
        if len(self.jump_times) > 0:
            last_jump_t = self.jump_times[-1]["duration_s"]
            self.panel.add(f"Pace: {last_jump_t:.2f}s/jump", COLOR_WHITE)

    def draw_custom(self, frame, frame_num):
        """Draw reference line on ball and alert if asymmetric."""
        h, w = frame.shape[:2]

        # Draw ball reference line
        if self.ball_x is not None:
            bx = int(self.ball_x)
            cv2.line(frame, (bx, int(h*0.3)), (bx, int(h*0.9)), (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "CENTER", (bx - 30, int(h*0.28)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Alert visualization
        if self.is_asymmetric_now:
            cv2.putText(frame, "Fix Knee Symmetry!", (w // 2 - 120, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        return frame

    def generate_report(self) -> dict:
        duration = 0
        if self.start_frame and self.end_frame:
            duration = (self.end_frame - self.start_frame) / self.fps

        # ── Pace Consistency ──
        # Same logic as Diamond Drill
        times = [j["duration_s"] for j in self.jump_times]
        avg_time = sum(times) / len(times) if times else 0
        std_time = float(np.std(times)) if len(times) > 1 else 0
        cv = (std_time / avg_time * 100) if avg_time > 0 else 0
        
        consistency_score = max(0, min(100, 100 - cv)) if times else 0

        slow_jumps = []
        fast_jumps = []
        for j in self.jump_times:
            if avg_time == 0: continue
            dev = (j["duration_s"] - avg_time) / avg_time * 100
            if dev > 35:
                slow_jumps.append({
                    "jump_num": j["jump_num"],
                    "duration_s": j["duration_s"],
                    "deviation_percent": round(dev, 1)
                })
            elif dev < -35:
                fast_jumps.append({
                    "jump_num": j["jump_num"],
                    "duration_s": j["duration_s"],
                    "deviation_percent": round(dev, 1)
                })

        feedback = []
        if cv < 15 and self.jump_count > 0:
            feedback.append("Great pace consistency! You maintained a steady rhythm.")
        elif self.jump_count > 0:
            feedback.append("Inconsistent rhythm. Try to keep your jump durations even.")

        pace_metrics = {
            "average_time_per_jump_s": round(avg_time, 2),
            "std_dev_s": round(std_time, 2),
            "coefficient_of_variation_percent": round(cv, 1),
            "consistency_score": round(consistency_score, 1),
            "slow_jumps_count": len(slow_jumps),
            "fast_jumps_count": len(fast_jumps),
            "feedback": feedback
        }

        # ── Form Assessment ──
        # Reduce form score based on asymmetry events
        # We group events by jump_num so a single jump doesn't drain the score entirely
        bad_jumps = len(set(e["jump_num"] for e in self.asymmetry_events))
        form_score = 100 if self.jump_count == 0 else max(0, 100 - (bad_jumps / max(1, self.jump_count) * 100))
        
        form_feedback = "Good Form: Both knees matched during jumps."
        if bad_jumps > 0:
            form_feedback = f"Warning: Asymmetric knee bend detected on {bad_jumps} out of {self.jump_count} jumps."

        form_metrics = {
            "symmetry_score": round(form_score, 1),
            "asymmetric_jumps_count": bad_jumps,
            "total_asymmetry_frames": len(self.asymmetry_events),
            "feedback": form_feedback
        }

        # ── Overall Score ──
        # 40% Volume (Jump count target ~ 20 jumps in 15s)
        # 30% Form (Symmetry)
        # 30% Consistency
        
        base_target_jumps = 20  # Arbitrary "perfect" score target for 15s
        volume_score = min(100, (self.jump_count / base_target_jumps) * 100)
        
        overall = volume_score * 0.4 + form_score * 0.3 + consistency_score * 0.3 if self.jump_count > 0 else 0

        return {
            "drill_info": {
                "time_analyzed_s": round(duration, 2)
            },
            "jump_metrics": {
                "total_jumps": self.jump_count,
                "volume_score": round(volume_score, 1)
            },
            "pace_consistency": pace_metrics,
            "form_assessment": form_metrics,
            "overall_score": round(overall, 1)
        }
