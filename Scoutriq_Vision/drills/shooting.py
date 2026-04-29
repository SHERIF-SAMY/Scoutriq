"""
shooting.py — Shooting Drill Analyzer (Redesigned).

State-machine approach with 3 states:
  WAITING → BALL_FLYING → COOLDOWN → WAITING → ...

Metrics per shot:
  - foot used (Left/Right)
  - inside gate (bool)
  - goal scored (bool)
  - goal zone value (0-10) with label
  - shot speed (m/s)
  - miss distance (m) — if missed

Overall metrics:
  - total drill time, touches, goals, accuracy, errors
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional
from collections import deque

from ..base_drill import BaseDrillAnalyzer
from ..config import DrillConfig
from ..constants import KP_L_ANKLE, KP_R_ANKLE
from ..core.geometry import box_center
from ..core.ball_physics import BallVelocityTracker

# ═══════════════════════════════════════════════════════════
#  GOAL ZONE GRID — 3 rows × 4 columns
#  Top corners (10) = hardest for keeper
#  Bottom center (2) = easiest
# ═══════════════════════════════════════════════════════════
GOAL_ZONE_VALUES = [
    [10, 7, 7, 10],   # top
    [ 5, 3, 3,  5],   # middle
    [ 5, 2, 2,  5],   # bottom
]
GOAL_ZONE_LABELS = [
    ["top-left",    "top-center-left",    "top-center-right",    "top-right"],
    ["mid-left",    "mid-center-left",    "mid-center-right",    "mid-right"],
    ["bottom-left", "bottom-center-left", "bottom-center-right", "bottom-right"],
]


class ShootingDrillAnalyzer(BaseDrillAnalyzer):
    """Analyzer for the shooting drill (3 shots through cone gate)."""

    drill_name = "Shooting Drill"

    # States
    ST_WAITING  = "WAITING"
    ST_FLYING   = "FLYING"
    ST_COOLDOWN = "COOLDOWN"

    def __init__(self, config: DrillConfig):
        config.drill_params.setdefault("track_goals", True)
        super().__init__(config)

        p = {**{
            "max_flight_seconds": 3.0,
            "cooldown_seconds": 1.5,
            "shot_speed_threshold_ms": 4.0,
            "kick_confirm_frames": 10,
            "touch_proximity_m": 0.25,
            "touch_cooldown_seconds": 0.3,
            "velocity_spike_threshold": 3.0,
            "direction_change_threshold_deg": 30.0,
        }, **config.drill_params}

        self._max_flight_sec   = float(p["max_flight_seconds"])
        self._cooldown_sec     = float(p["cooldown_seconds"])
        self._shot_speed_thr   = float(p["shot_speed_threshold_ms"])
        self._kick_confirm_n   = int(p["kick_confirm_frames"])
        self._touch_prox_m     = float(p["touch_proximity_m"])
        self._touch_cd_sec     = float(p["touch_cooldown_seconds"])
        self._vel_spike_thr    = float(p["velocity_spike_threshold"])
        self._dir_change_thr   = float(p["direction_change_threshold_deg"])

    # ═══════════════════════════════════════════════════════════
    #  SETUP
    # ═══════════════════════════════════════════════════════════

    def setup(self) -> None:
        # Gate / cones
        self._cone_positions: dict[int, tuple] = {}
        self._gate_polygon: Optional[np.ndarray] = None

        # Goal
        self._goal_box: Optional[tuple] = None
        self._last_known_goal_box: Optional[tuple] = None

        # Ball
        self._all_balls: list[dict] = []
        self._tracked_ball: Optional[np.ndarray] = None
        self._last_det_pos: Optional[np.ndarray] = None
        self._ball_history: deque = deque(maxlen=120)

        # Feet
        self._foot_L: Optional[np.ndarray] = None
        self._foot_R: Optional[np.ndarray] = None

        # State machine
        self._state = self.ST_WAITING
        self._state_frame = 0

        # Kick candidate (recorded while WAITING)
        self._near = False
        self._near_frame = 0
        self._near_pos: Optional[np.ndarray] = None
        self._near_foot_name: Optional[str] = None
        self._near_inside_gate = False

        # Shots & flight tracking
        self.shots: list[dict] = []
        self._flight_pts: list[tuple] = []

        # Touches
        self.touch_count = 0
        self._touch_cd = 0
        self.ball_velocity_tracker = BallVelocityTracker()

        # Timing
        self._t_start: Optional[int] = None
        self._t_end: Optional[int] = None

        # Errors
        self.errors: list[dict] = []

        # Excluded zones — positions of old balls (from previous shots)
        # that settled in/near the goal. These must be ignored.
        self._excluded_zones: list[np.ndarray] = []

        # Drawing helpers
        self._draw_ball: Optional[np.ndarray] = None
        self._draw_foot: Optional[np.ndarray] = None
        self._draw_foot_name: Optional[str] = None

    # ═══════════════════════════════════════════════════════════
    #  HOOKS
    # ═══════════════════════════════════════════════════════════

    def on_object_detected(self, frame_num, class_name, box, stable_id, confidence):
        cn = class_name.lower()
        if cn in self.config.ball_class_names:
            cx, cy = box_center(box)
            self._all_balls.append({"pos": np.array([cx, cy]), "box": box})
        elif cn in self.config.cone_class_names:
            cx, cy = box_center(box)
            self._cone_positions[stable_id] = (cx, cy, box)
        elif cn in self.config.goal_class_names:
            self._goal_box = tuple(float(v) for v in box[:4])
            self._last_known_goal_box = self._goal_box

    def on_pose_estimated(self, frame_num, keypoints, player_box, track_id):
        kp, thr = keypoints, self.config.keypoint_confidence
        if kp[KP_L_ANKLE][2] > thr:
            self._foot_L = np.array(kp[KP_L_ANKLE][:2])
        if kp[KP_R_ANKLE][2] > thr:
            self._foot_R = np.array(kp[KP_R_ANKLE][:2])

    # ═══════════════════════════════════════════════════════════
    #  PER-FRAME LOGIC
    # ═══════════════════════════════════════════════════════════

    def compute_drill_metrics(self, frame_num) -> None:
        self._rebuild_gate()

        # 1 — Ball selection
        ball_info = self._select_ball()
        if ball_info:
            self._tracked_ball = ball_info["pos"].copy()
            self._last_det_pos = self._tracked_ball.copy()
            self._ball_history.append((frame_num, self._tracked_ball.copy()))
            self._draw_ball = self._tracked_ball.copy()
        else:
            self._tracked_ball = None
            self._draw_ball = None
            if self._state != self.ST_FLYING:
                pass  # keep last_det_pos for continuity

        # 2 — Foot-ball info
        ball = self._tracked_ball
        dist_m, foot_name, foot_pos = self._foot_info(ball)
        self._draw_foot = foot_pos
        self._draw_foot_name = foot_name

        # 2b — Update ball velocity tracker
        ball_pos_tuple = tuple(ball) if ball is not None else None
        self.ball_velocity_tracker.update(frame_num, ball_pos_tuple)

        # 3 — Touch counting (with velocity validation)
        physics_touch = self.ball_velocity_tracker.has_physical_touch(
            velocity_threshold=self._vel_spike_thr,
            direction_threshold_deg=self._dir_change_thr,
        )
        if dist_m is not None and dist_m < self._touch_prox_m and self._touch_cd <= 0 and physics_touch:
            self.touch_count += 1
            self._touch_cd = int(self.fps * self._touch_cd_sec)
            if self._t_start is None:
                self._t_start = frame_num
        if self._touch_cd > 0:
            self._touch_cd -= 1

        # 4 — State machine
        if self._state == self.ST_WAITING:
            self._do_waiting(frame_num, ball, dist_m, foot_name)
        elif self._state == self.ST_FLYING:
            self._do_flying(frame_num, ball)
        elif self._state == self.ST_COOLDOWN:
            self._do_cooldown(frame_num)

        # 5 — Reset per-frame data
        self._all_balls = []
        self._goal_box = None
        self._foot_L = None
        self._foot_R = None

    # ─── STATE HANDLERS ───────────────────────────────────────

    def _do_waiting(self, fn, ball, dist_m, foot_name):
        """Detect kick: ball near foot → ball flies away fast."""
        if ball is None:
            return

        # Phase 1: record "near foot" candidate
        if dist_m is not None and dist_m < self._touch_prox_m:
            self._near = True
            self._near_frame = fn
            self._near_pos = ball.copy()
            self._near_foot_name = foot_name
            self._near_inside_gate = self._point_in_gate(ball)

        # Phase 2: check if ball flew away → kick confirmed
        if self._near and self._near_pos is not None:
            dt_frames = fn - self._near_frame
            if 2 <= dt_frames <= self._kick_confirm_n:
                disp = float(np.linalg.norm(ball - self._near_pos))
                dt_sec = dt_frames / max(self.fps, 1)
                cal = self.calibration
                dist = cal.px_to_m(disp) if cal.is_calibrated else disp / 250.0
                speed = dist / max(dt_sec, 0.001)

                if speed > self._shot_speed_thr:
                    # ✅ KICK!
                    shot = {
                        "shot_number": len(self.shots) + 1,
                        "foot": self._near_foot_name,
                        "inside_gate": self._near_inside_gate,
                        "frame_kick": self._near_frame,
                    }
                    self.shots.append(shot)

                    if not self._near_inside_gate:
                        self.errors.append({
                            "type": "shot_outside_gate",
                            "shot_number": shot["shot_number"],
                            "frame": self._near_frame,
                        })

                    self._state = self.ST_FLYING
                    self._state_frame = fn
                    self._flight_pts = [(fn, ball.copy())]
                    self._near = False
                    return

            if dt_frames > self._kick_confirm_n:
                self._near = False

    def _do_flying(self, fn, ball):
        """Track ball in flight, check for goal or timeout."""
        if ball is not None:
            self._flight_pts.append((fn, ball.copy()))

        goal = self._goal_box or self._last_known_goal_box
        shot = self.shots[-1] if self.shots else None
        if not shot or not goal:
            return

        # Check ALL non-excluded balls against goal (not just tracked)
        # This catches goals even when our tracked ball is wrong
        goal_ball_pos = self._find_new_ball_in_goal(goal)
        if goal_ball_pos is not None:
            self._resolve_goal(shot, fn, goal_ball_pos, goal)
            return

        # Timeout → but check one last time for ball in goal before declaring miss
        if fn - self._state_frame > int(self.fps * self._max_flight_sec):
            shot["goal"] = False
            shot["frame_resolved"] = fn
            shot["shot_speed_ms"] = self._calc_speed()
            shot["miss_distance_m"] = self._calc_miss_dist()
            self._t_end = fn
            self.errors.append({
                "type": "missed_shot",
                "shot_number": shot["shot_number"],
                "frame": fn,
                "miss_distance_m": shot.get("miss_distance_m"),
            })
            # Still add any ball near goal to excluded (it might be there but not "in" goal)
            self._exclude_balls_near_goal(goal)
            self._go_cooldown(fn)

    def _do_cooldown(self, fn):
        if fn - self._state_frame > int(self.fps * self._cooldown_sec):
            self._state = self.ST_WAITING
            self._state_frame = fn
            self._last_det_pos = None

    def _go_cooldown(self, fn):
        self._state = self.ST_COOLDOWN
        self._state_frame = fn
        self._tracked_ball = None
        self._last_det_pos = None
        self._near = False

    def _resolve_goal(self, shot, fn, ball_pos, goal):
        """Mark shot as goal and add ball to excluded zones."""
        shot["goal"] = True
        zv, zl = self._goal_zone(ball_pos, goal)
        shot["goal_zone_value"] = zv
        shot["goal_zone_label"] = zl
        shot["frame_resolved"] = fn
        shot["shot_speed_ms"] = self._calc_speed()
        self._t_end = fn
        # Exclude this ball position so future shots don't pick it up
        self._excluded_zones.append(ball_pos.copy())
        self._go_cooldown(fn)

    def _find_new_ball_in_goal(self, goal):
        """Check if any NON-EXCLUDED ball is inside the goal box."""
        gx1, gy1, gx2, gy2 = goal
        pad = 30
        for b in self._all_balls:
            pos = b["pos"]
            if (gx1 - pad) <= pos[0] <= (gx2 + pad) and \
               (gy1 - pad) <= pos[1] <= (gy2 + pad):
                if not self._is_excluded(pos):
                    return pos
        return None

    def _exclude_balls_near_goal(self, goal):
        """Add any ball near the goal to excluded zones (for future shots)."""
        gx1, gy1, gx2, gy2 = goal
        pad = 80  # generous padding
        for b in self._all_balls:
            pos = b["pos"]
            if (gx1 - pad) <= pos[0] <= (gx2 + pad) and \
               (gy1 - pad) <= pos[1] <= (gy2 + pad):
                if not self._is_excluded(pos):
                    self._excluded_zones.append(pos.copy())

    def _is_excluded(self, pos):
        """Check if a ball position is near any excluded zone (~1m radius)."""
        cal = self.calibration
        thr = cal.m_to_px(1.0) if cal.is_calibrated else 100
        for ep in self._excluded_zones:
            if float(np.linalg.norm(pos - ep)) < thr:
                return True
        return False

    # ─── HELPERS ──────────────────────────────────────────────

    def _select_ball(self):
        """Smart ball selection — state-aware.
        WAITING/COOLDOWN: continuity → gate → player proximity (strict)
        FLYING: very lenient — accept any ball, prefer closest to goal
        """
        if not self._all_balls:
            return None

        cal = self.calibration

        # ── FLYING STATE: lenient selection (but exclude dead balls) ──
        if self._state == self.ST_FLYING:
            live_balls = [b for b in self._all_balls if not self._is_excluded(b["pos"])]
            if not live_balls:
                return None

            # Strategy 1: Continuity with HUGE max jump (ball is fast)
            if self._last_det_pos is not None:
                best_d, best = float("inf"), None
                for b in live_balls:
                    d = float(np.linalg.norm(b["pos"] - self._last_det_pos))
                    if d < best_d:
                        best_d, best = d, b
                max_j = cal.m_to_px(15.0) if cal.is_calibrated else 3000
                if best_d <= max_j:
                    return best

            # Strategy 2: Closest live ball to goal
            goal = self._goal_box or self._last_known_goal_box
            if goal:
                gcx = (goal[0] + goal[2]) / 2
                gcy = (goal[1] + goal[3]) / 2
                goal_c = np.array([gcx, gcy])
                best_d, best = float("inf"), None
                for b in live_balls:
                    d = float(np.linalg.norm(b["pos"] - goal_c))
                    if d < best_d:
                        best_d, best = d, b
                return best

            # Strategy 3: Any live ball
            return live_balls[0]

        # ── WAITING / COOLDOWN: strict selection ──

        # A: Continuity (max 3m jump)
        if self._last_det_pos is not None:
            best_d, best = float("inf"), None
            for b in self._all_balls:
                d = float(np.linalg.norm(b["pos"] - self._last_det_pos))
                if d < best_d:
                    best_d, best = d, b
            max_j = cal.m_to_px(3.0) if cal.is_calibrated else 600
            if best_d <= max_j:
                return best

        # B: Gate
        for b in self._all_balls:
            if self._point_in_gate(b["pos"]):
                return b

        # C: Proximity to feet
        best_d, best = float("inf"), None
        for b in self._all_balls:
            dL = float(np.linalg.norm(b["pos"] - self._foot_L)) if self._foot_L is not None else float("inf")
            dR = float(np.linalg.norm(b["pos"] - self._foot_R)) if self._foot_R is not None else float("inf")
            d = min(dL, dR)
            thr = cal.m_to_px(1.5) if cal.is_calibrated else 400
            if d < thr and d < best_d:
                best_d, best = d, b
        return best

    def _foot_info(self, ball):
        """Returns (distance_m, foot_name, foot_pos) or (None, None, None)."""
        if ball is None or (self._foot_L is None and self._foot_R is None):
            return None, None, None

        dL = float(np.linalg.norm(ball - self._foot_L)) if self._foot_L is not None else float("inf")
        dR = float(np.linalg.norm(ball - self._foot_R)) if self._foot_R is not None else float("inf")
        d_px = min(dL, dR)
        cal = self.calibration
        d_m = cal.px_to_m(d_px) if cal.is_calibrated else d_px / 250.0

        if dL < dR:
            return d_m, "Left", self._foot_L
        return d_m, "Right", self._foot_R

    def _goal_zone(self, pos, goal_box):
        """Map ball position to goal zone value + label."""
        gx1, gy1, gx2, gy2 = goal_box
        gw, gh = max(gx2 - gx1, 1), max(gy2 - gy1, 1)

        rx = max(0.0, min(1.0, (pos[0] - gx1) / gw))
        ry = max(0.0, min(1.0, (pos[1] - gy1) / gh))

        col = min(int(rx * 4), 3)
        row = min(int(ry * 3), 2)
        return GOAL_ZONE_VALUES[row][col], GOAL_ZONE_LABELS[row][col]

    def _calc_speed(self):
        """Shot speed from first flight positions (m/s)."""
        if len(self._flight_pts) < 2:
            return 0.0
        n = min(len(self._flight_pts), 5)
        f1, p1 = self._flight_pts[0]
        f2, p2 = self._flight_pts[n - 1]
        d_px = float(np.linalg.norm(p2 - p1))
        dt = (f2 - f1) / max(self.fps, 1)
        if dt <= 0:
            return 0.0
        cal = self.calibration
        d_m = cal.px_to_m(d_px) if cal.is_calibrated else d_px / 250.0
        return round(d_m / dt, 2)

    def _calc_miss_dist(self):
        """Min distance from any flight position to goal edge (metres)."""
        goal = self._last_known_goal_box
        if not goal or not self._flight_pts:
            return None
        gx1, gy1, gx2, gy2 = goal
        best = float("inf")
        for _, pos in self._flight_pts:
            cx = max(gx1, min(gx2, pos[0]))
            cy = max(gy1, min(gy2, pos[1]))
            d = float(np.linalg.norm(pos - np.array([cx, cy])))
            best = min(best, d)
        if best >= float("inf"):
            return None
        cal = self.calibration
        return round(cal.px_to_m(best) if cal.is_calibrated else best / 250.0, 2)

    # ─── GATE ─────────────────────────────────────────────────

    def _rebuild_gate(self):
        if len(self._cone_positions) < 4:
            return
        sc = sorted(self._cone_positions.values(), key=lambda c: c[0])
        lp = sorted(sc[:2], key=lambda c: c[1])
        rp = sorted(sc[2:], key=lambda c: c[1])
        pts = [
            (int(lp[0][0]), int(lp[0][2][3])),
            (int(rp[0][0]), int(rp[0][2][3])),
            (int(rp[1][0]), int(rp[1][2][3])),
            (int(lp[1][0]), int(lp[1][2][3])),
        ]
        self._gate_polygon = np.array(pts, np.int32).reshape((-1, 1, 2))

    def _point_in_gate(self, point) -> bool:
        if self._gate_polygon is None or point is None:
            return False
        return cv2.pointPolygonTest(
            self._gate_polygon, (int(point[0]), int(point[1])), False
        ) >= 0

    # ═══════════════════════════════════════════════════════════
    #  OVERLAY & DRAWING
    # ═══════════════════════════════════════════════════════════

    def build_overlay(self, frame_num) -> None:
        self.panel.clear()
        goals  = sum(1 for s in self.shots if s.get("goal"))
        misses = sum(1 for s in self.shots if s.get("goal") is False)

        self.panel.add(f"State: {self._state}", (200, 200, 200))
        self.panel.add(f"Touches: {self.touch_count}", (0, 0, 255))
        self.panel.add(f"Shots: {len(self.shots)}/3", (255, 200, 0))
        clr = (0, 200, 0) if goals >= misses else (0, 0, 255)
        self.panel.add(f"Goals: {goals} | Misses: {misses}", clr)

        if self.shots:
            s = self.shots[-1]
            if s.get("goal") is True:
                zv = s.get("goal_zone_value", "?")
                self.panel.add(f"Last: GOAL (Zone: {zv})", (0, 255, 0))
            elif s.get("goal") is False:
                md = s.get("miss_distance_m", "?")
                self.panel.add(f"Last: MISS ({md}m away)", (0, 0, 255))
            else:
                self.panel.add("Last: Tracking...", (255, 255, 0))

        if self._t_start:
            elapsed = (frame_num - self._t_start) / max(self.fps, 1)
            self.panel.add(f"Time: {elapsed:.1f}s", (255, 255, 255))

    def draw_custom(self, frame, frame_num) -> np.ndarray:
        # Gate overlay
        if self._gate_polygon is not None:
            cv2.polylines(frame, [self._gate_polygon], True, (0, 200, 255), 3)
            ov = frame.copy()
            cv2.fillPoly(ov, [self._gate_polygon], (0, 150, 255))
            cv2.addWeighted(ov, 0.22, frame, 0.78, 0, frame)

        # Active ball
        if self._draw_ball is not None:
            pt = tuple(self._draw_ball.astype(int))
            clr = (0, 255, 255) if self._state == self.ST_FLYING else (0, 255, 0)
            cv2.circle(frame, pt, 15, clr, 3)
            cv2.drawMarker(frame, pt, (0, 0, 0), cv2.MARKER_CROSS, 8, 1)
            if self._draw_foot is not None:
                fp = tuple(self._draw_foot.astype(int))
                cv2.line(frame, pt, fp, (0, 100, 255), 2)

        # Goal zone grid
        goal = self._goal_box or self._last_known_goal_box
        if goal:
            self._draw_goal_grid(frame, goal)

        return frame

    def _draw_goal_grid(self, frame, gb):
        gx1, gy1, gx2, gy2 = [int(v) for v in gb]
        gw, gh = gx2 - gx1, gy2 - gy1
        for i in range(1, 4):
            x = gx1 + int(gw * i / 4)
            cv2.line(frame, (x, gy1), (x, gy2), (255, 255, 255), 1)
        for i in range(1, 3):
            y = gy1 + int(gh * i / 3)
            cv2.line(frame, (gx1, y), (gx2, y), (255, 255, 255), 1)
        for row in range(3):
            for col in range(4):
                val = GOAL_ZONE_VALUES[row][col]
                cx = gx1 + int(gw * (col + 0.5) / 4)
                cy = gy1 + int(gh * (row + 0.5) / 3)
                cv2.putText(frame, str(val), (cx - 8, cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    # ═══════════════════════════════════════════════════════════
    #  REPORT
    # ═══════════════════════════════════════════════════════════

    def generate_report(self) -> dict:
        goals = sum(1 for s in self.shots if s.get("goal"))
        n = len(self.shots)
        acc = (goals / max(n, 1)) * 100
        inside = sum(1 for s in self.shots if s.get("inside_gate", False))

        if self._t_start and self._t_end:
            drill_time = (self._t_end - self._t_start) / max(self.fps, 1)
        else:
            drill_time = self.total_frames / max(self.fps, 1)

        zone_vals = [s.get("goal_zone_value", 0) for s in self.shots if s.get("goal")]
        avg_zone = sum(zone_vals) / len(zone_vals) if zone_vals else 0

        # Score: Accuracy 50% + Zone quality 30% + Gate compliance 20%
        score = round(
            acc * 0.5
            + (avg_zone / 10) * 30
            + (inside / max(n, 1)) * 20,
            1,
        )

        return {
            "drill_info": {"drill_type": "Shooting Drill"},
            "shooting_metrics": {
                "total_time_s": round(drill_time, 2),
                "total_touches": self.touch_count,
                "total_shots": n,
                "goals_scored": goals,
                "shots_missed": n - goals,
                "accuracy_pct": round(acc, 1),
                "shots_inside_gate": inside,
                "shots_outside_gate": n - inside,
                "avg_goal_zone_value": round(avg_zone, 1),
            },
            "shots": self.shots,
            "errors": self.errors,
            "overall_score": score,
        }
