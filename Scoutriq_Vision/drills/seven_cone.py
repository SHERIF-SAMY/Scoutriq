"""
seven_cone.py — 7-Cone Dribble Drill Analyzer.

Analyzes the 7-cone dribble drill from video:
  - 7 cones in a straight line, 1m apart
  - Player dribbles one ball through all cones using both feet,
    turns around at the end, and dribbles back.

Measurements:
  • Total time
  • Number of touches (total + per-foot)
  • Ball separation distance from foot
  • Number of times ball touched cones (cone contacts)

Errors flagged:
  • Missed cone path
  • Cone contact (ball hits a cone)
  • Loss of close control (ball > 1m from foot)
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from ..base_drill import BaseDrillAnalyzer
from ..config import DrillConfig
from ..constants import KP_L_ANKLE, KP_R_ANKLE, COLOR_CONE, COLOR_GOOD, COLOR_ERROR
from ..core.geometry import dist2d, box_center
from ..core.ball_physics import BallVelocityTracker

import cv2


# Default drill params (overridable via config.drill_params)
_DEFAULTS = {
    "num_cones": 7,
    "cone_spacing_m": 1.0,
    "close_control_threshold_m": 1.0,
    "touch_distance_threshold_m": 0.35,
    "touch_cooldown_frames": 8,
    "cone_contact_distance_m": 0.15,
    "velocity_spike_threshold": 3.0,
    "direction_change_threshold_deg": 30.0,
}


class SevenConeDrillAnalyzer(BaseDrillAnalyzer):
    """Analyzer for the 7-cone dribble drill."""

    drill_name = "7-Cone Dribble"

    def __init__(self, config: DrillConfig):
        super().__init__(config)
        p = {**_DEFAULTS, **config.drill_params}
        self.num_cones: int = p["num_cones"]
        self.cone_spacing_m: float = p["cone_spacing_m"]
        self.close_control_m: float = p["close_control_threshold_m"]
        self.touch_dist_m: float = p["touch_distance_threshold_m"]
        self.touch_cooldown: int = p["touch_cooldown_frames"]
        self.cone_contact_dist_m: float = p["cone_contact_distance_m"]
        self._vel_spike_thr: float = p["velocity_spike_threshold"]
        self._dir_change_thr: float = p["direction_change_threshold_deg"]

    def setup(self) -> None:
        # ── Time ──
        self.start_frame: Optional[int] = None
        self.end_frame: Optional[int] = None

        # ── Cones ──
        self.cone_positions: dict[int, tuple[float, float]] = {}
        self.cone_boxes: dict[int, tuple] = {}
        self.ordered_cone_ids: list[int] = []

        # ── Ball ──
        self.ball_position: Optional[tuple[float, float]] = None
        self.ball_velocity_tracker = BallVelocityTracker()

        # ── Touches ──
        self.total_touches: int = 0
        self.left_foot_touches: int = 0
        self.right_foot_touches: int = 0
        self.last_touch_frame: int = -999
        self._prev_ball_near: bool = False

        # ── Ball separation ──
        self.ball_foot_distances: list[float] = []

        # ── Cone contacts ──
        self.cone_contact_count: int = 0
        self.cone_contact_frames: list[tuple[int, int]] = []
        self._cone_cooldown: dict[int, int] = {}

        # ── Cone path ──
        self.cone_visit_order: list[int] = []
        self.missed_cones: list[int] = []

        # ── Loss of control ──
        self.loss_of_control_count: int = 0
        self.loss_of_control_frames: list[int] = []
        self._loc_cooldown: int = -999

        # ── Player ──
        self.smoothed_pos: Optional[tuple[float, float]] = None

    # ── hooks ──

    def on_object_detected(self, frame_num, class_name, box, stable_id, confidence):
        if class_name.lower() in self.config.ball_class_names:
            self.ball_position = box_center(box)
        else:
            center = box_center(box)
            self.cone_positions[stable_id] = center
            self.cone_boxes[stable_id] = tuple(box)

    def on_pose_estimated(self, frame_num, keypoints, player_box, track_id):
        raw = box_center(player_box)
        if self.start_frame is None:
            self.start_frame = frame_num
        self.end_frame = frame_num

        alpha = self.config.position_smoothing
        if self.smoothed_pos is None:
            self.smoothed_pos = raw
        else:
            self.smoothed_pos = (
                alpha * raw[0] + (1 - alpha) * self.smoothed_pos[0],
                alpha * raw[1] + (1 - alpha) * self.smoothed_pos[1],
            )

    def compute_drill_metrics(self, frame_num):
        self._order_cones_once()
        kp = self.current_keypoints
        ppm = self.calibration.pixels_per_meter or 200

        # ── Update ball velocity tracker ──
        self.ball_velocity_tracker.update(frame_num, self.ball_position)

        # ── Touch detection ──
        if self.ball_position is not None and kp is not None and len(kp) > 16:
            l_ankle, r_ankle = kp[KP_L_ANKLE], kp[KP_R_ANKLE]
            l_dist = dist2d(self.ball_position, (l_ankle[0], l_ankle[1])) if l_ankle[2] > 0.3 else 9999
            r_dist = dist2d(self.ball_position, (r_ankle[0], r_ankle[1])) if r_ankle[2] > 0.3 else 9999
            min_dist = min(l_dist, r_dist)
            self.ball_foot_distances.append(min_dist)

            thresh = self.touch_dist_m * ppm
            ball_near = min_dist < thresh
            physics_touch = self.ball_velocity_tracker.has_physical_touch(
                velocity_threshold=self._vel_spike_thr,
                direction_threshold_deg=self._dir_change_thr,
            )
            if ball_near and not self._prev_ball_near and physics_touch:
                if (frame_num - self.last_touch_frame) >= self.touch_cooldown:
                    self.total_touches += 1
                    foot = "left" if l_dist < r_dist else "right"
                    if foot == "left":
                        self.left_foot_touches += 1
                    else:
                        self.right_foot_touches += 1
                    self.last_touch_frame = frame_num
            self._prev_ball_near = ball_near

            # Loss of control
            control_px = self.close_control_m * ppm
            if min_dist > control_px:
                if (frame_num - self._loc_cooldown) > self.fps * 0.5:
                    self.loss_of_control_count += 1
                    self.loss_of_control_frames.append(frame_num)
                    self._loc_cooldown = frame_num
                self.current_errors.append("LOST CONTROL")

        # ── Cone contact ──
        if self.ball_position is not None:
            contact_px = self.cone_contact_dist_m * ppm
            for cid, cpos in self.cone_positions.items():
                if dist2d(self.ball_position, cpos) < contact_px:
                    last = self._cone_cooldown.get(cid, -999)
                    if (frame_num - last) > self.fps * 1.0:
                        self.cone_contact_count += 1
                        self.cone_contact_frames.append((frame_num, cid))
                        self._cone_cooldown[cid] = frame_num
                    self.current_errors.append("CONE CONTACT")

        # ── Cone path ──
        if self.ball_position is not None and self.ordered_cone_ids:
            nearest, nearest_d = None, float("inf")
            for cid in self.ordered_cone_ids:
                d = dist2d(self.ball_position, self.cone_positions[cid])
                if d < nearest_d:
                    nearest_d = d
                    nearest = cid
            visit_px = self.cone_spacing_m * 0.6 * ppm
            if nearest is not None and nearest_d < visit_px:
                idx = self.ordered_cone_ids.index(nearest)
                if not self.cone_visit_order or self.cone_visit_order[-1] != idx:
                    if self.cone_visit_order:
                        prev_idx = self.cone_visit_order[-1]
                        step = 1 if idx > prev_idx else -1
                        for m in range(prev_idx + step, idx, step):
                            if m not in self.cone_visit_order:
                                self.missed_cones.append(m)
                    self.cone_visit_order.append(idx)

        if self.missed_cones:
            self.current_errors.append("MISSED CONE")

    def build_overlay(self, frame_num):
        self.panel.clear()
        ppm = self.calibration.pixels_per_meter

        # Time
        if self.start_frame is not None:
            elapsed = (frame_num - self.start_frame) / self.fps
            self.panel.add(f"Time: {elapsed:.1f}s")
        else:
            self.panel.add("Time: --")

        # Touches
        self.panel.add(f"Touches: {self.total_touches}  "
                       f"(L:{self.left_foot_touches} R:{self.right_foot_touches})")

        # Ball distance
        if self.ball_foot_distances:
            recent = self.ball_foot_distances[-15:]
            avg_px = sum(recent) / len(recent)
            dist_m = self.calibration.px_to_m(avg_px)
            color = COLOR_GOOD if dist_m < 1.0 else COLOR_ERROR
            self.panel.add(f"Ball Dist: {dist_m:.2f} m", color)

        # Cone contacts
        cc_color = COLOR_ERROR if self.cone_contact_count > 0 else COLOR_GOOD
        self.panel.add(f"Cone Contacts: {self.cone_contact_count}", cc_color)

        # Cones detected
        self.panel.add(f"Cones: {len(self.cone_positions)}/{self.num_cones}")

        # Loss of control
        loc_color = COLOR_ERROR if self.loss_of_control_count > 0 else COLOR_GOOD
        self.panel.add(f"Control Lost: {self.loss_of_control_count}x", loc_color)

    def draw_custom(self, frame, frame_num):
        from ..visualization.drawing import draw_ball_foot_line

        # Ball–foot line
        if self.ball_position is not None and self.current_keypoints is not None:
            frame = draw_ball_foot_line(
                frame, self.ball_position, self.current_keypoints,
                px_to_m_func=self.calibration.px_to_m,
            )

        # Cone numbers
        if self.ordered_cone_ids:
            for idx, cid in enumerate(self.ordered_cone_ids):
                if cid in self.cone_positions:
                    cx, cy = int(self.cone_positions[cid][0]), int(self.cone_positions[cid][1])
                    label = f"C{idx + 1}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(frame, (cx - tw // 2 - 4, cy - 30 - th),
                                  (cx + tw // 2 + 4, cy - 26), COLOR_CONE, -1)
                    cv2.putText(frame, label, (cx - tw // 2, cy - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return frame

    def generate_report(self) -> dict:
        self.calibration.compute()
        is_cal = self.calibration.is_calibrated

        duration = 0
        if self.start_frame and self.end_frame:
            duration = (self.end_frame - self.start_frame) / self.fps

        avg_sep_m, max_sep_m, close_pct = 0, 0, 0
        if self.ball_foot_distances:
            dists_m = [self.calibration.px_to_m(d) for d in self.ball_foot_distances]
            avg_sep_m = float(np.mean(dists_m))
            max_sep_m = float(np.max(dists_m))
            close_frames = sum(1 for d in dists_m if d <= self.close_control_m)
            close_pct = (close_frames / len(dists_m)) * 100

        touch_score = min(100, (self.total_touches / max(duration, 1)) * 20) if duration > 0 else 0
        ball_score = max(0, min(100, 100 - (avg_sep_m - 0.2) * 200)) if avg_sep_m > 0 else 50
        error_penalty = (self.cone_contact_count * 10
                         + len(self.missed_cones) * 15
                         + self.loss_of_control_count * 5)
        overall = max(0, min(100,
                             touch_score * 0.3 + ball_score * 0.4
                             + max(0, 100 - error_penalty) * 0.3))

        return {
            "drill_info": {},
            "time_metrics": {"total_duration_seconds": round(duration, 2)},
            "touch_metrics": {
                "total_touches": self.total_touches,
                "left_foot_touches": self.left_foot_touches,
                "right_foot_touches": self.right_foot_touches,
                "touches_per_second": round(self.total_touches / max(duration, 0.01), 2),
            },
            "ball_control": {
                "average_separation_meters": round(avg_sep_m, 3),
                "max_separation_meters": round(max_sep_m, 3),
                "close_control_percentage": round(close_pct, 1),
                "ball_control_score": round(ball_score, 1),
            },
            "errors": {
                "cone_contacts": self.cone_contact_count,
                "missed_cones": len(self.missed_cones),
                "loss_of_control_count": self.loss_of_control_count,
            },
            "cone_path": {"visit_order": [int(c) for c in self.cone_visit_order]},
            "overall_score": round(overall, 1),
        }

    # ── internal ──

    def _order_cones_once(self):
        if len(self.ordered_cone_ids) >= self.num_cones:
            return
        if len(self.cone_positions) >= self.num_cones:
            xs = [p[0] for p in self.cone_positions.values()]
            ys = [p[1] for p in self.cone_positions.values()]
            x_range = max(xs) - min(xs)
            y_range = max(ys) - min(ys)
            key_func = (lambda cid: self.cone_positions[cid][0]) if x_range >= y_range \
                   else (lambda cid: self.cone_positions[cid][1])
            self.ordered_cone_ids = sorted(self.cone_positions.keys(), key=key_func)
            layout = "horizontal" if x_range >= y_range else "vertical"
            print(f"  ✓ Cones ordered ({len(self.ordered_cone_ids)} cones): {layout} layout")
