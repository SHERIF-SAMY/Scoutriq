"""
t_test.py — T-Test Agility Drill Analyzer.

Setup: 4 cones in a T shape (A=Start, B=Middle, C=Left, D=Right).
Sequence: A -> B -> C -> D -> B -> A.
Metrics: 
- Total time
- Time for each section (A-B, B-C, C-D, D-B, B-A)
- Acceleration / Deceleration rate
- Time lost at change of direction
Errors: Knocking cone, stopping before turning, cone not reached.
"""

from __future__ import annotations

import math
import cv2
import numpy as np
from typing import Optional
from collections import deque

from ..base_drill import BaseDrillAnalyzer
from ..config import DrillConfig
from ..core.geometry import dist2d, box_center
from ..constants import COLOR_GOOD, COLOR_ERROR, KP_L_ANKLE, KP_R_ANKLE, KP_L_SHOULDER, KP_R_SHOULDER, KP_L_HIP, KP_R_HIP

COLOR_WHITE = (255, 255, 255)
COLOR_WARNING = (0, 165, 255)

_DEFAULTS = {
    # Distance in meters to consider 'reached' a cone
    "cone_reach_threshold_m": 1.0,
    # Minimum speed (m/s) to drop below to consider "changing direction"
    "cod_speed_threshold": 1.5,
    # Time in seconds below speed threshold to be considered "stopped" (error)
    "stop_error_seconds": 1.5,
    # Speed smoothing window
    "speed_smoothing": 5
}

class TTestAnalyzer(BaseDrillAnalyzer):
    drill_name = "T-Test"

    def __init__(self, config: DrillConfig):
        super().__init__(config)
        p = {**_DEFAULTS, **config.drill_params}
        self.reach_thresh_m = p["cone_reach_threshold_m"]
        self.cod_speed_thresh = p["cod_speed_threshold"]
        self.stop_error_sec = p["stop_error_seconds"]
        self.speed_smooth_window = p["speed_smoothing"]
        
        # ── State ──
        self.start_frame: Optional[int] = None
        self.end_frame: Optional[int] = None
        
        # Cone tracking
        # We need to map 4 cones: start(A), mid(B), left(C), right(D).
        # We'll initially just store all detected cones and map them when we have 4.
        self.cones: dict[int, tuple[float, float]] = {}  # id -> (x,y)
        self.cone_boxes: dict[int, tuple] = {}           # id -> (x1,y1,x2,y2)
        self.mapped_cones: dict[str, int] = {}           # "A" -> id
        self.H: Optional[np.ndarray] = None              # 2D Map Homography
        
        # T-Test Official Distances (meters)
        self.OFFICIAL_DISTANCES = {
            "A-B": 9.14,  # 10 yards
            "B-D": 4.57,  # 5 yards
            "D-C": 9.14,  # 10 yards (across)
            "C-B": 4.57,  # 5 yards
            "B-A": 9.14   # 10 yards
        }
        
        # Player tracking
        self.player_path_m: list[tuple[float, float]] = []
        self.last_player_pos: Optional[tuple[float, float]] = None
        
        # Live overlay speed tracking
        self.current_speed: float = 0.0  
        self.speed_history: deque[float] = deque(maxlen=15) # Smooth over 0.5s
        
        # Sequence tracking
        # A -> B -> D -> C -> B -> A (User requested sequence)
        self.expected_sequence = ["A", "B", "D", "C", "B", "A"]
        self.current_waypoint_idx = 0
        self.section_times: dict[str, float] = {}       # "A-B": 2.5s
        self.section_speeds: dict[str, float] = {}      # Average speed per section
        
        self.last_waypoint_frame: Optional[int] = None
        self.min_dist_to_target: float = float('inf')
        self.min_dist_frame: Optional[int] = None
        
        # Change of Direction (COD) tracking (Dwell time at cones)
        self.cod_events: dict[str, float] = {}  # cone -> dwell time
        self.current_dwell_start: Optional[int] = None
        
        # Errors
        self.errors = {
            "cone_not_reached": 0,
            "stopped_before_turning": 0,
            "knocked_cone": 0,
            "wrong_sequence": 0
        }
        
        # Form Analysis (Side Shuffle vs Forward Run in Lateral Sections)
        self.lateral_sections = ["B-D", "D-C", "C-B"]
        self.form_total_frames: dict[str, int] = {k: 0 for k in self.lateral_sections}
        self.form_bad_frames: dict[str, int] = {k: 0 for k in self.lateral_sections}
        self.is_bad_form = False
        
        self.active_alert: Optional[str] = None
        self.alert_frames = 0
        
    def setup(self) -> None:
        pass

    def _map_cones(self):
        """Map the 4 detected cones to A (start), B (mid), C (left), D (right)."""
        if len(self.cones) != 4 or len(self.mapped_cones) == 4:
            return
            
        # Sort cones by Y (bottom to top). Larger Y = closer to camera (bottom of image)
        # Assuming front view: Start cone (A) is at the top (smallest Y) or bottom (largest Y).
        # We'll assume the user is starting from bottom and running UP, 
        # or starting UP and running DOWN. Let's look at the image:
        # In the front view, A is far away (small Y, top of screen). 
        # C, B, D are closer (large Y, bottom of screen).
        
        items = list(self.cones.items())
        # Sort by Y ascending (top to bottom)
        items.sort(key=lambda x: x[1][1])
        
        # The top-most cone is Start (A)
        id_A, pos_A = items[0]
        
        # The remaining 3 are C, B, D. They share roughly the same Y.
        # Sort them by X to get Left (C), Mid (B), Right (D).
        bottom_three = items[1:]
        bottom_three.sort(key=lambda x: x[1][0])
        
        id_C = bottom_three[0][0]
        id_B = bottom_three[1][0]
        id_D = bottom_three[2][0]
        
        self.mapped_cones = {
            "A": id_A,
            "C": id_C,
            "B": id_B,
            "D": id_D
        }
        print(f"Mapped cones: Start(A)={id_A}, Mid(B)={id_B}, Left(C)={id_C}, Right(D)={id_D}")
        self._build_homography()

    def _build_homography(self):
        if self.H is not None or len(self.mapped_cones) != 4:
            return
            
        iA = self.cones[self.mapped_cones["A"]]
        iC = self.cones[self.mapped_cones["C"]]
        iD = self.cones[self.mapped_cones["D"]]
        iB = self.cones[self.mapped_cones["B"]]
        
        boxA = self.cone_boxes[self.mapped_cones["A"]]
        boxB = self.cone_boxes[self.mapped_cones["B"]]
        hA = boxA[3] - boxA[1]
        hB = boxB[3] - boxB[1]
        
        # A is far away in the front view, so hA < hB. scale_factor < 1
        scale_factor = max(0.1, hA / hB) if hB > 0 else 0.5
        
        W10 = math.hypot(iD[0] - iC[0], iD[1] - iC[1])
        W0 = W10 * scale_factor
        
        angle = math.atan2(iD[1] - iC[1], iD[0] - iC[0])
        
        iA_left = (iA[0] - (W0/2) * math.cos(angle), iA[1] - (W0/2) * math.sin(angle))
        iA_right = (iA[0] + (W0/2) * math.cos(angle), iA[1] + (W0/2) * math.sin(angle))
        
        # Real-world coords (meters)
        # A_left = (-5, 0), A_right = (5, 0), D = (5, 10), C = (-5, 10)
        src_pts = np.array([iA_left, iA_right, iD, iC], dtype=np.float32)
        dst_pts = np.array([[-5, 0], [5, 0], [5, 10], [-5, 10]], dtype=np.float32)
        
        self.H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        print("2D Map Homography Built Successfully!")

    def _to_2d_m(self, px_pos: tuple[float, float]) -> Optional[tuple[float, float]]:
        if self.H is None:
            return None
        pt = np.array([[[px_pos[0], px_pos[1]]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, self.H)
        return (mapped[0][0][0], mapped[0][0][1])

    def on_object_detected(self, frame_num, class_name, box, stable_id, confidence):
        if class_name.lower() in self.config.cone_class_names:
            center = box_center(box)
            self.cone_boxes[stable_id] = box
            
            # Update cone position continuously (to catch knocks)
            if stable_id in self.cones:
                old_center = self.cones[stable_id]
                dist = dist2d(old_center, center)
                
                # If cone moved > 30 pixels suddenly, it might be knocked
                if dist > 30 and len(self.mapped_cones) == 4:
                    # Only map error if player is very close (use px dist since map might jump if knocked)
                    if self.player_path_m and self.last_player_pos:
                        if dist2d(self.last_player_pos, old_center) < ((self.calibration.pixels_per_meter or 1.0) * 1.5):
                            self._trigger_alert("Cone Knocked!")
                            self.errors["knocked_cone"] += 1
                            
            self.cones[stable_id] = center

    def on_pose_estimated(self, frame_num, keypoints, player_box, track_id):
        self._map_cones()
        
        # Base pos is average of ankles
        l_ankle = keypoints[KP_L_ANKLE]
        r_ankle = keypoints[KP_R_ANKLE]
        l_shoulder = keypoints[KP_L_SHOULDER]
        r_shoulder = keypoints[KP_R_SHOULDER]
        l_hip = keypoints[KP_L_HIP]
        r_hip = keypoints[KP_R_HIP]
        
        conf = self.config.keypoint_confidence
        
        if l_ankle[2] > conf and r_ankle[2] > conf:
            p_pos = ((l_ankle[0] + r_ankle[0]) / 2, (l_ankle[1] + r_ankle[1]) / 2)
        else:
            p_pos = (player_box[0] + player_box[2] / 2, player_box[3])
            
        self.last_player_pos = p_pos
        
        p_m = self._to_2d_m(p_pos)
        self.player_path_m.append(p_m)
        
        # Determine current section
        current_section = None
        if self.start_frame and not self.end_frame and self.current_waypoint_idx < len(self.expected_sequence):
             current_section = f"{self.expected_sequence[self.current_waypoint_idx-1]}-{self.expected_sequence[self.current_waypoint_idx]}"
        
        # Calculate strict 2D map speed for trigger logic and overlay
        speed = 0.0
        if len(self.player_path_m) >= 2:
            dist_m = dist2d(self.player_path_m[-2], p_m)
            speed = dist_m * self.fps  # m/s
            
        # Smooth the live speed for the overlay
        if self.start_frame and not self.end_frame:
            # Cap raw speed to 8.0 m/s to prevent tracking glitches from spiking the average
            capped_speed = min(speed, 8.0)
            self.speed_history.append(capped_speed)
            self.current_speed = sum(self.speed_history) / len(self.speed_history)
        else:
            self.current_speed = 0.0

        # Form Analysis for Side Shuffle
        self.is_bad_form = False
        if current_section in self.lateral_sections and speed > 1.0:
            if l_shoulder[2] > conf and r_shoulder[2] > conf and l_hip[2] > conf and r_hip[2] > conf:
                # If facing forward, shoulders and hips are wide on X-axis.
                # If facing sideways (running forward), shoulders overlap on X-axis.
                shoulder_dx = abs(l_shoulder[0] - r_shoulder[0])
                hip_dx = abs(l_hip[0] - r_hip[0])
                
                # Height of torso as a reference scale
                torso_h = abs(((l_shoulder[1] + r_shoulder[1])/2) - ((l_hip[1] + r_hip[1])/2))
                
                if torso_h > 0:
                    shoulder_ratio = shoulder_dx / torso_h
                    hip_ratio = hip_dx / torso_h
                    
                    # If shoulders are extremely narrow relative to body length, player is turned sideways.
                    if shoulder_ratio < 0.4 and hip_ratio < 0.4:
                        self.is_bad_form = True
                        self.form_bad_frames[current_section] += 1
                        self._trigger_alert("Improper Form (Face Forward!)")
                
                self.form_total_frames[current_section] += 1
                        
        # Waypoint tracking in 2D Space
        if self.current_waypoint_idx < len(self.expected_sequence):
            target_wp = self.expected_sequence[self.current_waypoint_idx]
            target_id = self.mapped_cones[target_wp]
            target_pos_px = self.cones[target_id]
            target_pos_m = self._to_2d_m(target_pos_px)
            
            dist_to_target = dist2d(p_m, target_pos_m)
            
            # Start timer when moving away from A
            if self.current_waypoint_idx == 0:
                if dist_to_target > 1.0 and speed > 1.0:
                    self.start_frame = frame_num
                    self.last_waypoint_frame = frame_num
                    self.current_waypoint_idx = 1
                    self.min_dist_to_target = float('inf')
                    self.min_dist_frame = frame_num
            else:
                if dist_to_target < self.min_dist_to_target:
                    self.min_dist_to_target = dist_to_target
                    self.min_dist_frame = frame_num
                
                # COD Dwell time logic: if within 1.5m of target, start dwell timer
                if dist_to_target <= 1.5 and self.current_dwell_start is None:
                    self.current_dwell_start = frame_num
                    
                # Arrived at target waypoint
                if dist_to_target <= self.reach_thresh_m:
                    section_name = f"{self.expected_sequence[self.current_waypoint_idx-1]}-{target_wp}"
                    time_taken = (frame_num - self.last_waypoint_frame) / self.fps
                    self.section_times[section_name] = time_taken
                    
                    # Calculate real speed
                    if section_name in self.OFFICIAL_DISTANCES and time_taken > 0:
                        self.section_speeds[section_name] = self.OFFICIAL_DISTANCES[section_name] / time_taken
                    
                    # Record COD dwell
                    if self.current_dwell_start:
                        dwell_time = (frame_num - self.current_dwell_start) / self.fps
                        if dwell_time > self.stop_error_sec:
                            self.errors["stopped_before_turning"] += 1
                            self._trigger_alert("Stopped before turn!")
                        self.cod_events[target_wp] = dwell_time
                    
                    self.last_waypoint_frame = frame_num
                    self.current_waypoint_idx += 1
                    self.min_dist_to_target = float('inf')
                    self.min_dist_frame = frame_num
                    self.current_dwell_start = None
                    
                    if self.current_waypoint_idx == len(self.expected_sequence):
                        self.end_frame = frame_num
                        
                # Abandoned target (turned around and moved away by 1.5m from closest point)
                elif dist_to_target > self.min_dist_to_target + 1.5:
                    self.errors["cone_not_reached"] += 1
                    self._trigger_alert(f"Missed {target_wp}!")
                    
                    # Record distance up to the exact turnaround moment (min_dist_frame)
                    section_name = f"{self.expected_sequence[self.current_waypoint_idx-1]}-{target_wp}"
                    time_taken = (self.min_dist_frame - self.last_waypoint_frame) / self.fps
                    self.section_times[section_name] = time_taken
                    
                    if section_name in self.OFFICIAL_DISTANCES and time_taken > 0:
                        self.section_speeds[section_name] = self.OFFICIAL_DISTANCES[section_name] / time_taken
                    
                    if self.current_dwell_start:
                        dwell_time_up_to_turn = (self.min_dist_frame - self.current_dwell_start) / self.fps
                        self.cod_events[target_wp] = max(0, dwell_time_up_to_turn)
                    
                    # Advance to the NEXT waypoint starting exactly from the turnaround frame
                    self.last_waypoint_frame = self.min_dist_frame
                    self.current_waypoint_idx += 1
                    self.min_dist_to_target = float('inf')
                    self.min_dist_frame = frame_num
                    self.current_dwell_start = None
                    
                    if self.current_waypoint_idx == len(self.expected_sequence):
                        self.end_frame = frame_num

    def _trigger_alert(self, msg: str):
        self.active_alert = msg
        self.alert_frames = int(self.fps * 2)  # show for 2s

    def compute_drill_metrics(self, frame_num):
        pass
        
    def build_overlay(self, frame_num):
        self.panel.clear()
        
        # Title
        self.panel.add("T-TEST", COLOR_WHITE)
        
        # Timer
        if self.start_frame:
            end = self.end_frame or frame_num
            elapsed = (end - self.start_frame) / self.fps
            self.panel.add(f"Time: {elapsed:.2f}s", COLOR_GOOD)
            
        # Target
        if self.current_waypoint_idx < len(self.expected_sequence):
            target = self.expected_sequence[self.current_waypoint_idx]
            if target == "A": target = "Start (A)"
            elif target == "B": target = "Middle (B)"
            elif target == "C": target = "Left (C)"
            elif target == "D": target = "Right (D)"
            self.panel.add(f"Target: {target}", COLOR_WARNING)
        else:
            self.panel.add("FINISHED!", COLOR_GOOD)
            
        # Active Speed (only if drill is running)
        if self.start_frame and not self.end_frame:
            self.panel.add(f"Live Speed: {self.current_speed:.1f} m/s", COLOR_WHITE)
            
        # Sections Display
        for k, v in self.section_times.items():
            s_speed = self.section_speeds.get(k, 0.0)
            self.panel.add(f"{k}: {v:.2f}s | Avg Dist Speed {s_speed:.1f} m/s", COLOR_WHITE)
            
    def draw_custom(self, frame, frame_num):
        # Draw cones with labels
        for label, cid in self.mapped_cones.items():
            if cid in self.cones:
                cx, cy = map(int, self.cones[cid])
                cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
                cv2.putText(frame, label, (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
        # Draw target cone highlight
        if self.current_waypoint_idx < len(self.expected_sequence):
            target_wp = self.expected_sequence[self.current_waypoint_idx]
            if target_wp in self.mapped_cones:
                cid = self.mapped_cones[target_wp]
                if cid in self.cones:
                    cx, cy = map(int, self.cones[cid])
                    cv2.circle(frame, (cx, cy), 15, (0, 165, 255), 2)
                    
        # Alert handling
        if self.active_alert and self.alert_frames > 0:
            h, w = frame.shape[:2]
            cv2.putText(frame, self.active_alert, (w//2 - 150, 100), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)
            self.alert_frames -= 1
        elif self.alert_frames <= 0:
            self.active_alert = None
            
        return frame

    def generate_report(self) -> dict:
        total_time = 0.0
        if self.start_frame and self.end_frame:
            total_time = (self.end_frame - self.start_frame) / self.fps
            
        # Compile strictly per-section metrics
        sections_data = {}
        max_speed_achieved = 0.0
        
        for k, time_taken in self.section_times.items():
            avg_speed = self.section_speeds.get(k, 0.0)
            max_speed_achieved = max(max_speed_achieved, avg_speed)
            dist_m = self.OFFICIAL_DISTANCES.get(k, 0.0)
            
            # Simple average acceleration for the section (a = v/t) assuming start from 0
            # Deceleration is assumed symmetric for simplicity in a pure distance model
            avg_accel = avg_speed / (time_taken / 2) if time_taken > 0 else 0
            
            section_info = {
                "distance_m": dist_m,
                "time_s": round(time_taken, 2),
                "avg_speed_m_s": round(avg_speed, 2),
                "avg_accel_m_s2": round(avg_accel, 2)
            }
            
            # Add form score for lateral sections
            if k in self.lateral_sections:
                total_f = self.form_total_frames.get(k, 0)
                bad_f = self.form_bad_frames.get(k, 0)
                form_score = 100
                if total_f > 0:
                    good_ratio = max(0.0, 1.0 - (bad_f / total_f))
                    form_score = int(good_ratio * 100)
                section_info["side_shuffle_form_score"] = form_score
                
            sections_data[k] = section_info
        
        # Calculate COD overall stats
        avg_cod_time = sum(self.cod_events.values()) / len(self.cod_events) if self.cod_events else 0
        
        # Calculate overall form penalty
        total_lat_frames = sum(self.form_total_frames.values())
        total_bad_frames = sum(self.form_bad_frames.values())
        form_penalty = 0
        if total_lat_frames > 0:
            bad_ratio = total_bad_frames / total_lat_frames
            # up to 20 points penalty for terrible form
            form_penalty = int(bad_ratio * 20.0)
        
        # Calculate scores
        # Baseline good T-Test is < 10.0 seconds
        time_score = 100 if total_time <= 9.0 else max(0, 100 - (total_time - 9.0) * 10)
        time_score = total_time > 0 and time_score or 0
        
        error_penalty = (sum(self.errors.values()) * 10) + form_penalty
        overall = max(0, time_score - error_penalty)
        
        return {
            "drill_info": {
                "drill_type": "T-Test",
                "completed": self.end_frame is not None,
                "total_time_s": round(total_time, 2)
            },
            "sections": sections_data,
            "overall_kinematics": {
                "max_section_speed_m_s": round(max_speed_achieved, 2)
            },
            "change_of_direction_dwell_times": {k: round(v, 2) for k, v in self.cod_events.items()},
            "overall_change_of_direction": {
                "avg_dwell_time_s": round(avg_cod_time, 2),
                "total_cod_events": len(self.cod_events)
            },
            "errors": self.errors,
            "scoring": {
                "time_score": round(time_score, 1),
                "error_penalty": sum(self.errors.values()) * 10,
                "form_penalty": form_penalty,
                "overall_score": round(overall, 1)
            }
        }
