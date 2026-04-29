"""
diamond.py — Diamond Cone Sprint Drill Analyzer.

Uses HOMOGRAPHY calibration (not scalar px/m) because the diamond drill
has a fixed geometric formation of 4+1 cones with known 4m spacing.

Speed is computed via WAYPOINT tracking: known 4m between cones, measured
transit time → accurate speed without pixel noise.

Drill journey: S → L → B → R → T → L → S  (6 legs × 4m = 24m total)
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional

from ..base_drill import BaseDrillAnalyzer
from ..config import DrillConfig
from ..constants import (
    KP_L_SHOULDER, KP_R_SHOULDER, KP_L_HIP, KP_R_HIP,
    KP_L_ANKLE, KP_R_ANKLE, COLOR_GOOD, COLOR_ERROR,
)
from ..core.geometry import dist2d, box_center
from ..core.homography import HomographyCalibrator
from ..core.waypoint_tracker import WaypointTracker

# Default drill params
_DEFAULTS = {
    "cone_spacing_m": 4.0,
    "min_movement_px": 3.0,
    "journey": ['S', 'L', 'B', 'R', 'T', 'L', 'S'],
}

# Cone label map for display
CONE_LABEL_MAP = {'T': 'Top', 'R': 'Right', 'B': 'Bottom', 'L': 'Left', 'S': 'Start'}

# Colors
COLOR_CONE_DEFAULT = (0, 255, 255)    # Yellow
COLOR_CONE_REACHED = (0, 255, 0)      # Green
COLOR_CONE_TARGET  = (0, 0, 255)      # Red
COLOR_TARGET_LINE  = (0, 100, 255)    # Orange


class DiamondDrillAnalyzer(BaseDrillAnalyzer):
    """Analyzer for the diamond cone sprint drill.

    Uses homography calibration (perspective transform) and waypoint-based
    speed computation for accurate real-world metrics.
    """

    drill_name = "Diamond Drill"

    def __init__(self, config: DrillConfig):
        super().__init__(config)
        p = {**_DEFAULTS, **config.drill_params}
        self.cone_spacing_m: float = p["cone_spacing_m"]
        self.min_movement_px: float = p["min_movement_px"]
        self.journey: list[str] = p["journey"]

    def setup(self) -> None:
        # ── Homography calibration ──
        self.homography = HomographyCalibrator(cone_spacing=self.cone_spacing_m)
        self._homography_ready = False
        self._first_frame_cones: list[tuple[float, float]] = []
        self._collecting_cones = True  # collect cone positions on first pass

        # ── Waypoint tracking ──
        self.waypoint: Optional[WaypointTracker] = None

        # ── Time ──
        self.start_frame: Optional[int] = None
        self.end_frame: Optional[int] = None

        # ── Player position / speed ──
        self.smoothed_pos: Optional[tuple[float, float]] = None
        self.player_positions_smoothed: list[tuple[int, float, float]] = []
        self.player_speeds: list[float] = []

        # ── Ball ──
        self.ball_positions: list[tuple[float, float]] = []
        self.ball_player_distances: list[float] = []

        # ── Cones (tracked every frame for drawing) ──
        self.cone_positions: dict[int, tuple[float, float]] = {}

        # ── Pose metrics ──
        self.balance_scores: list[float] = []
        self.center_of_gravity: list[float] = []

    # ── hooks ──

    def on_object_detected(self, frame_num, class_name, box, stable_id, confidence):
        center = box_center(box)

        if class_name.lower() in self.config.ball_class_names:
            self.ball_positions.append(center)
            self._update_ball_distance(center)
        else:
            # Cone
            self.cone_positions[stable_id] = center
            # Collect for homography on first frame
            if self._collecting_cones and frame_num <= 3:
                self._first_frame_cones.append(center)

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

        self.player_positions_smoothed.append(
            (frame_num, self.smoothed_pos[0], self.smoothed_pos[1])
        )

        # Speed from smoothed positions
        if len(self.player_positions_smoothed) >= 2:
            prev = self.player_positions_smoothed[-2]
            curr = self.player_positions_smoothed[-1]
            d = dist2d((prev[1], prev[2]), (curr[1], curr[2]))
            self.player_speeds.append(d if d >= self.min_movement_px else 0)

        # Pose metrics
        self._update_pose_metrics(keypoints)

    def compute_drill_metrics(self, frame_num):
        # ── Initialize homography on first frames ──
        if self._collecting_cones and frame_num == 3:
            self._collecting_cones = False
            # Use ALL cone positions collected so far (across first few frames)
            all_cones = list(self.cone_positions.values())
            if len(all_cones) >= 4:
                H = self.homography.calibrate_from_cones(all_cones)
                if H is not None:
                    self._homography_ready = True
                    print(f"   ✓ Homography calibration: {len(all_cones)} cones detected")

                    # Initialize waypoint tracker
                    if self.homography.labeled_cones:
                        self.waypoint = WaypointTracker(
                            self.journey,
                            self.homography.labeled_cones,
                            self.cone_spacing_m,
                            self.fps,
                        )
                        print(f"   ✓ Waypoint tracker: journey {' → '.join(self.journey)}")
                else:
                    print(f"   ⚠ Homography failed — falling back to standard calibration")

        # ── Waypoint tracking ──
        if self.waypoint and self.smoothed_pos:
            self.waypoint.update(self.smoothed_pos, frame_num)

        # ── Cone proximity ──
        if self.smoothed_pos and self.cone_positions:
            min_d = min(dist2d(self.smoothed_pos, pos) for pos in self.cone_positions.values())
            if self._homography_ready:
                # Convert to meters via homography (more accurate)
                pass  # Store in pixels, convert in report
            # Store raw pixel distance for now
            # (converted to meters at report time using whatever calibration is active)

    def build_overlay(self, frame_num):
        self.panel.clear()
        SMOOTH = 15

        # Waypoint-based info
        if self.waypoint:
            wp = self.waypoint
            legs_done = wp.legs_completed
            dist_m = wp.total_distance
            last_speed_kmh = wp.completed_legs[-1]['speed_kmh'] if wp.completed_legs else 0.0
            target = wp.current_target() or "DONE"
            target_name = CONE_LABEL_MAP.get(target, target)

            self.panel.add(f"Leg Spd: {last_speed_kmh:.1f} km/h", COLOR_GOOD)
            self.panel.add(f"Distance: {dist_m:.0f} m", (255, 255, 255))
            self.panel.add(f"Legs: {legs_done}/{len(wp.journey)-1}", (255, 255, 255))
            self.panel.add(f"Next: {target_name}", COLOR_CONE_TARGET)
        else:
            # Fallback: pixel-based speed
            if self.player_speeds:
                recent = self.player_speeds[-SMOOTH:]
                avg_ppf = sum(recent) / len(recent) * self.fps
                cal = self.calibration
                if cal.is_calibrated:
                    speed_m = cal.px_to_m(avg_ppf)
                    self.panel.add(f"Speed: {speed_m:.1f} m/s", COLOR_GOOD)
                else:
                    self.panel.add(f"Speed: {avg_ppf:.0f} px/s", COLOR_GOOD)

        # Ball distance
        if self.ball_player_distances:
            recent = self.ball_player_distances[-SMOOTH:]
            avg_d = sum(recent) / len(recent)
            cal = self.calibration
            if cal.is_calibrated:
                self.panel.add(f"Ball: {cal.px_to_m(avg_d):.2f} m", (255, 255, 255))
            elif self._homography_ready and self.ball_positions and self.smoothed_pos:
                try:
                    d_m = self.homography.world_distance(self.ball_positions[-1], self.smoothed_pos)
                    self.panel.add(f"Ball: {d_m:.2f} m", (255, 255, 255))
                except Exception:
                    self.panel.add(f"Ball: {avg_d:.0f} px", (255, 255, 255))

        # Balance
        if self.balance_scores:
            recent = self.balance_scores[-SMOOTH:]
            avg_b = sum(recent) / len(recent) * 100
            self.panel.add(f"Balance: {avg_b:.0f}%", COLOR_GOOD)

    def draw_custom(self, frame, frame_num):
        """Draw cone markers with journey progress + line to target."""
        # Map stable_id cones to labels using proximity to initial labeled positions
        live_labeled = self._get_live_labeled_cones()

        if live_labeled:
            for label, pos in live_labeled.items():
                cx, cy = int(pos[0]), int(pos[1])
                # Color: reached / target / default
                target = self.waypoint.current_target() if self.waypoint else None
                if label == target:
                    color = COLOR_CONE_TARGET
                elif self.waypoint and label in [
                    self.journey[j] for j in range(self.waypoint.current_target_idx)
                ]:
                    color = COLOR_CONE_REACHED
                else:
                    color = COLOR_CONE_DEFAULT

                cv2.circle(frame, (cx, cy), 5, color, -1)
                cv2.circle(frame, (cx, cy), 6, (255, 255, 255), 1)
                full_name = CONE_LABEL_MAP.get(label, label)
                cv2.putText(frame, full_name, (cx - 12, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

            # Line from player to target cone (live position)
            if self.smoothed_pos and self.waypoint:
                target = self.waypoint.current_target()
                if target and target in live_labeled:
                    tp = live_labeled[target]
                    cv2.line(frame,
                             (int(self.smoothed_pos[0]), int(self.smoothed_pos[1])),
                             (int(tp[0]), int(tp[1])),
                             COLOR_TARGET_LINE, 2)

        # Ball-foot line
        from ..visualization.drawing import draw_ball_foot_line
        if self.ball_positions and self.current_keypoints is not None:
            px_to_m = self.calibration.px_to_m if self.calibration.is_calibrated else None
            frame = draw_ball_foot_line(
                frame, self.ball_positions[-1], self.current_keypoints,
                px_to_m_func=px_to_m,
            )

        return frame

    def generate_report(self) -> dict:
        self.calibration.compute()
        is_cal = self.calibration.is_calibrated
        ppm = self.calibration.pixels_per_meter

        report: dict = {
            "drill_info": {},
            "time_metrics": {},
            "speed_metrics": {},
            "ball_control": {},
            "movement_quality": {},
            "cone_metrics": {},
            "waypoint_data": {},
            "overall_score": 0,
        }

        # Calibration info
        report["drill_info"]["calibration_method"] = (
            "homography" if self._homography_ready else
            ("ball_diameter" if is_cal else "none")
        )

        # Time
        if self.start_frame and self.end_frame:
            dur = (self.end_frame - self.start_frame) / self.fps
            report["time_metrics"] = {
                "total_duration_seconds": round(dur, 2),
                "frames_analyzed": self.end_frame - self.start_frame,
            }

        # Speed — prefer waypoint data (most accurate)
        if self.waypoint and self.waypoint.completed_legs:
            wp = self.waypoint.summary()
            report["speed_metrics"] = {
                "average_speed_mps": wp['average_speed_mps'],
                "average_speed_kmh": wp['average_speed_kmh'],
                "max_speed_mps": wp['max_speed_mps'],
                "max_speed_kmh": wp['max_speed_kmh'],
                "total_distance_meters": wp['total_distance_meters'],
                "source": "waypoint",
                "unit": "meters",
            }
            report["waypoint_data"] = {
                "legs_completed": wp['legs_completed'],
                "legs_expected": wp['legs_expected'],
                "total_time_seconds": wp['total_time_seconds'],
                "legs": wp['legs'],
            }
        elif self.player_speeds:
            # Fallback: pixel-based speed
            avg_ppf = float(np.mean(self.player_speeds))
            max_ppf = float(np.max(self.player_speeds))
            total_px = float(sum(self.player_speeds))
            if is_cal:
                report["speed_metrics"] = {
                    "average_speed_mps": round(avg_ppf * self.fps / ppm, 2),
                    "max_speed_mps": round(max_ppf * self.fps / ppm, 2),
                    "total_distance_meters": round(total_px / ppm, 2),
                    "source": "pixel_displacement",
                    "unit": "meters",
                }

        # Ball control
        ball_control_score = 50
        if self.ball_player_distances:
            avg_d = float(np.mean(self.ball_player_distances))
            if is_cal:
                avg_m = avg_d / ppm
                close_px = 1.0 * ppm
                poss = sum(1 for d in self.ball_player_distances if d < close_px) / len(self.ball_player_distances) * 100
                ball_control_score = max(0, min(100, 100 - (avg_m - 0.5) * 100))
                report["ball_control"] = {
                    "average_distance_meters": round(avg_m, 2),
                    "possession_percentage": round(poss, 1),
                    "ball_control_score": round(ball_control_score, 1),
                }

        # Movement quality
        if self.balance_scores:
            avg_b = float(np.mean(self.balance_scores)) * 100
            report["movement_quality"]["balance_score"] = round(avg_b, 1)
        if self.center_of_gravity:
            avg_cog = float(np.mean(self.center_of_gravity))
            agility = min(100, (1 - avg_cog / self.frame_height) * 150)
            report["movement_quality"]["agility_score"] = round(agility, 1)

        # Cone metrics — use homography for distances if available
        cone_clearance = 50
        if self.cone_positions and self.player_positions_smoothed:
            min_cone_dists = []
            for _, px, py in self.player_positions_smoothed:
                min_d = min(dist2d((px, py), pos) for pos in self.cone_positions.values())
                min_cone_dists.append(min_d)
            avg_c_px = float(np.mean(min_cone_dists))
            min_c_px = float(np.min(min_cone_dists))

            if is_cal:
                avg_c = avg_c_px / ppm
                min_c = min_c_px / ppm
            elif self._homography_ready:
                # Approximate using homography scale
                avg_c = avg_c_px / (self.homography.avg_cone_dist_px / self.cone_spacing_m) if hasattr(self.homography, 'avg_cone_dist_px') else avg_c_px
                min_c = min_c_px / (self.homography.avg_cone_dist_px / self.cone_spacing_m) if hasattr(self.homography, 'avg_cone_dist_px') else min_c_px
            else:
                avg_c = avg_c_px
                min_c = min_c_px

            if min_c < 0.1:
                cone_clearance = 50
            elif min_c <= 0.5:
                cone_clearance = 100
            else:
                cone_clearance = max(60, 100 - (min_c - 0.5) * 80)

            report["cone_metrics"] = {
                "average_distance_meters": round(avg_c, 2),
                "closest_approach_meters": round(min_c, 2),
                "clearance_score": round(cone_clearance, 1),
                "cones_tracked": len(self.cone_positions),
            }

        # Pace consistency — are all legs roughly equal?
        if self.waypoint and len(self.waypoint.completed_legs) >= 2:
            legs = self.waypoint.completed_legs
            times = [leg['time_seconds'] for leg in legs]
            avg_time = sum(times) / len(times)
            std_time = float(np.std(times))
            cv = (std_time / avg_time * 100) if avg_time > 0 else 0  # coefficient of variation %

            # Consistency score: 100 = perfectly even, 0 = wildly inconsistent
            # CV of 0% = 100, CV of 50%+ = 0
            consistency_score = max(0, min(100, 100 - cv * 2))

            # Find slow and fast legs (>20% deviation from average)
            slow_legs = []
            fast_legs = []
            for leg in legs:
                deviation = (leg['time_seconds'] - avg_time) / avg_time * 100
                if deviation > 20:
                    slow_legs.append({
                        "leg": f"{leg['from']}→{leg['to']}",
                        "time_seconds": leg['time_seconds'],
                        "deviation_percent": round(deviation, 1),
                    })
                elif deviation < -20:
                    fast_legs.append({
                        "leg": f"{leg['from']}→{leg['to']}",
                        "time_seconds": leg['time_seconds'],
                        "deviation_percent": round(deviation, 1),
                    })

            # Build feedback
            feedback = []
            if cv < 15:
                feedback.append("Great pace consistency! Speed is well maintained across all legs.")
            else:
                feedback.append("Pace is inconsistent — try to maintain the same speed on every leg.")
                for sl in slow_legs:
                    feedback.append(
                        f"Slowed down on {sl['leg']} ({sl['time_seconds']:.2f}s, "
                        f"+{sl['deviation_percent']:.0f}% slower than average)"
                    )
                for fl in fast_legs:
                    feedback.append(
                        f"Fast on {fl['leg']} ({fl['time_seconds']:.2f}s, "
                        f"{abs(fl['deviation_percent']):.0f}% faster than average)"
                    )

            report["pace_consistency"] = {
                "average_leg_time_seconds": round(avg_time, 3),
                "std_dev_seconds": round(std_time, 3),
                "coefficient_of_variation_percent": round(cv, 1),
                "consistency_score": round(consistency_score, 1),
                "slow_legs": slow_legs,
                "fast_legs": fast_legs,
                "feedback": feedback,
            }

        # Overall score
        scores = []
        if "ball_control_score" in report.get("ball_control", {}):
            scores.append((report["ball_control"]["ball_control_score"], 0.30))
        if "balance_score" in report.get("movement_quality", {}):
            scores.append((report["movement_quality"]["balance_score"], 0.20))
        if "agility_score" in report.get("movement_quality", {}):
            scores.append((report["movement_quality"]["agility_score"], 0.15))
        if "clearance_score" in report.get("cone_metrics", {}):
            scores.append((report["cone_metrics"]["clearance_score"], 0.15))
        if "consistency_score" in report.get("pace_consistency", {}):
            scores.append((report["pace_consistency"]["consistency_score"], 0.20))
        if scores:
            ws = sum(s * w for s, w in scores) / sum(w for _, w in scores)
            report["overall_score"] = round(ws, 1)

        return report

    # ── internal ──

    def _update_ball_distance(self, ball_center):
        kp = self.current_keypoints
        if kp is not None and len(kp) > 16:
            l_a, r_a = kp[KP_L_ANKLE], kp[KP_R_ANKLE]
            dists = []
            if l_a[2] > 0.3:
                dists.append(dist2d(ball_center, (l_a[0], l_a[1])))
            if r_a[2] > 0.3:
                dists.append(dist2d(ball_center, (r_a[0], r_a[1])))
            if dists:
                self.ball_player_distances.append(min(dists))
                return
        if self.smoothed_pos:
            self.ball_player_distances.append(dist2d(ball_center, self.smoothed_pos))

    def _get_live_labeled_cones(self) -> dict[str, tuple[float, float]]:
        """Map live YOLO cone detections to journey labels via proximity
        to initial labeled positions. This way markers follow camera shake."""
        if not self.homography.labeled_cones or not self.cone_positions:
            return self.homography.labeled_cones  # fallback to initial

        live_labeled = {}
        used_ids = set()
        for label, initial_pos in self.homography.labeled_cones.items():
            best_id = None
            best_dist = float('inf')
            for sid, pos in self.cone_positions.items():
                if sid in used_ids:
                    continue
                d = dist2d(initial_pos, pos)
                if d < best_dist:
                    best_dist = d
                    best_id = sid
            if best_id is not None and best_dist < 150:  # max 150px drift
                live_labeled[label] = self.cone_positions[best_id]
                used_ids.add(best_id)
            else:
                live_labeled[label] = initial_pos  # fallback

        # Also update waypoint tracker with live positions
        if self.waypoint:
            self.waypoint.cone_px = live_labeled

        return live_labeled

    def _update_pose_metrics(self, keypoints):
        if len(keypoints) < 17:
            return
        conf = self.config.keypoint_confidence

        def get_kp(idx):
            if keypoints[idx][2] > conf:
                return (float(keypoints[idx][0]), float(keypoints[idx][1]))
            return None

        ls, rs = get_kp(KP_L_SHOULDER), get_kp(KP_R_SHOULDER)
        lh, rh = get_kp(KP_L_HIP), get_kp(KP_R_HIP)

        if all(p is not None for p in [ls, rs, lh, rh]):
            sc = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
            hc = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
            lateral = abs(sc[0] - hc[0])
            torso_h = abs(sc[1] - hc[1])
            if torso_h > 10:
                self.balance_scores.append(max(0, 1 - lateral / torso_h))
            self.center_of_gravity.append(hc[1])
