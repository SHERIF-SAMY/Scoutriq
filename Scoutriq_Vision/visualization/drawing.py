"""
drawing.py — Shared drawing functions for all drills.

Previously duplicated (identically) in diamond.py, seven_cone_dribble.py,
weakfoot.py, weekfootgpt.py, jump.py.  Now in one place.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..constants import (
    COCO_SKELETON_CONNECTIONS,
    MEDIAPIPE_SKELETON_CONNECTIONS,
    COLOR_SKELETON,
    COLOR_KEYPOINT,
    COLOR_GOOD,
    COLOR_ERROR,
    CLASS_COLORS,
    COLOR_DEFAULT,
    BOX_THICKNESS,
    FONT_SCALE,
    FONT_THICKNESS,
    LABEL_PADDING,
)
from ..core.geometry import dist2d


# ═══════════════════════════════════════════════════════════════
#  SKELETON DRAWING
# ═══════════════════════════════════════════════════════════════

def draw_pose(
    frame: np.ndarray,
    keypoints: np.ndarray,
    confidence_threshold: float = 0.2,
    skeleton_type: str = "coco",
    skeleton_color: tuple = COLOR_SKELETON,
    keypoint_color: tuple = COLOR_KEYPOINT,
) -> np.ndarray:
    """
    Draw skeleton and keypoints on a frame.

    Args:
        frame: BGR image.
        keypoints: (N, 3) array — [x, y, conf] per keypoint.
        confidence_threshold: Minimum confidence to draw.
        skeleton_type: 'coco' (17 kp) or 'mediapipe' (33 kp).
        skeleton_color: BGR color for bones.
        keypoint_color: BGR color for keypoints.

    Returns:
        The annotated frame.
    """
    connections = (
        COCO_SKELETON_CONNECTIONS
        if skeleton_type == "coco"
        else MEDIAPIPE_SKELETON_CONNECTIONS
    )

    # Draw bones
    for idx1, idx2 in connections:
        if idx1 < len(keypoints) and idx2 < len(keypoints):
            kp1, kp2 = keypoints[idx1], keypoints[idx2]
            if kp1[2] > confidence_threshold and kp2[2] > confidence_threshold:
                pt1 = (int(kp1[0]), int(kp1[1]))
                pt2 = (int(kp2[0]), int(kp2[1]))
                cv2.line(frame, pt1, pt2, skeleton_color, 2, cv2.LINE_AA)

    # Draw keypoints
    # Slightly larger radius for major joints (shoulders, hips)
    major_joints = {5, 6, 11, 12} if skeleton_type == "coco" else {11, 12, 23, 24}
    for i, kp in enumerate(keypoints):
        if kp[2] > confidence_threshold:
            x, y = int(kp[0]), int(kp[1])
            r = 5 if i in major_joints else 4
            cv2.circle(frame, (x, y), r, keypoint_color, -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), r, (0, 0, 0), 1, cv2.LINE_AA)

    return frame


# ═══════════════════════════════════════════════════════════════
#  BOUNDING BOX DRAWING
# ═══════════════════════════════════════════════════════════════

def draw_custom_box(
    frame: np.ndarray,
    box,
    label: str,
    class_id: int = 0,
    color: tuple | None = None,
) -> np.ndarray:
    """
    Draw a labeled bounding box.

    Args:
        frame: BGR image.
        box: (x1, y1, x2, y2).
        label: Text label to display (e.g. "ID:3 cone 95%").
        class_id: Used to pick color from CLASS_COLORS if color is None.
        color: Override color (BGR).

    Returns:
        The annotated frame.
    """
    x1, y1, x2, y2 = map(int, box)
    c = color or CLASS_COLORS.get(class_id, COLOR_DEFAULT)

    cv2.rectangle(frame, (x1, y1), (x2, y2), c, BOX_THICKNESS)

    (tw, th), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS
    )
    cv2.rectangle(
        frame,
        (x1, y1 - th - 2 * LABEL_PADDING),
        (x1 + tw + 2 * LABEL_PADDING, y1),
        c,
        -1,
    )
    cv2.putText(
        frame,
        label,
        (x1 + LABEL_PADDING, y1 - LABEL_PADDING),
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SCALE,
        (255, 255, 255),
        FONT_THICKNESS,
        cv2.LINE_AA,
    )
    return frame


# ═══════════════════════════════════════════════════════════════
#  BALL–FOOT LINE
# ═══════════════════════════════════════════════════════════════

def draw_ball_foot_line(
    frame: np.ndarray,
    ball_pos: tuple[float, float] | None,
    keypoints: np.ndarray | None,
    ankle_indices: tuple[int, int] = (15, 16),
    px_to_m_func=None,
    conf_threshold: float = 0.3,
) -> np.ndarray:
    """
    Draw a line from ball to nearest ankle, colored by distance.

    Colors:
        green  — < 0.5m (good control)
        orange — 0.5–1.0m (loose)
        red    — > 1.0m (lost control)

    Args:
        frame: BGR image.
        ball_pos: (cx, cy) ball position.
        keypoints: (N, 3) keypoints array.
        ankle_indices: Tuple of (left_ankle_idx, right_ankle_idx).
        px_to_m_func: Callable to convert pixel distance to metres.
        conf_threshold: Minimum keypoint confidence.

    Returns:
        The annotated frame.
    """
    if ball_pos is None or keypoints is None:
        return frame

    l_idx, r_idx = ankle_indices
    if l_idx >= len(keypoints) or r_idx >= len(keypoints):
        return frame

    # Find closest ankle
    best_foot = None
    best_dist = float("inf")
    for idx in [l_idx, r_idx]:
        ankle = keypoints[idx]
        if ankle[2] > conf_threshold:
            d = dist2d(ball_pos, (ankle[0], ankle[1]))
            if d < best_dist:
                best_dist = d
                best_foot = (int(ankle[0]), int(ankle[1]))

    if best_foot is None:
        return frame

    dist_m = px_to_m_func(best_dist) if px_to_m_func else best_dist

    if dist_m < 0.5:
        color = (0, 255, 0)        # green
    elif dist_m < 1.0:
        color = (0, 200, 255)      # orange
    else:
        color = (0, 0, 255)        # red

    bx, by = int(ball_pos[0]), int(ball_pos[1])
    cv2.line(frame, (bx, by), best_foot, color, 2, cv2.LINE_AA)

    # Distance label at midpoint
    mid = ((bx + best_foot[0]) // 2, (by + best_foot[1]) // 2)
    unit = "m" if px_to_m_func else "px"
    cv2.putText(
        frame,
        f"{dist_m:.2f}{unit}",
        mid,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )
    return frame
