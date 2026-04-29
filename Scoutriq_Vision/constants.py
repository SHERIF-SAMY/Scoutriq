"""
constants.py — Shared constants for the ScoutAI framework.

All skeleton definitions, keypoint indices, and drawing defaults in one place.
"""

# ═══════════════════════════════════════════════════════════════
#  COCO KEYPOINT INDICES  (used by YOLOv8-Pose, 17 keypoints)
# ═══════════════════════════════════════════════════════════════

KP_NOSE = 0
KP_L_EYE = 1
KP_R_EYE = 2
KP_L_EAR = 3
KP_R_EAR = 4
KP_L_SHOULDER = 5
KP_R_SHOULDER = 6
KP_L_ELBOW = 7
KP_R_ELBOW = 8
KP_L_WRIST = 9
KP_R_WRIST = 10
KP_L_HIP = 11
KP_R_HIP = 12
KP_L_KNEE = 13
KP_R_KNEE = 14
KP_L_ANKLE = 15
KP_R_ANKLE = 16


# ═══════════════════════════════════════════════════════════════
#  MEDIAPIPE KEYPOINT INDICES  (BlazePose, 33 keypoints)
# ═══════════════════════════════════════════════════════════════

MP_NOSE = 0
MP_L_SHOULDER = 11
MP_R_SHOULDER = 12
MP_L_ELBOW = 13
MP_R_ELBOW = 14
MP_L_WRIST = 15
MP_R_WRIST = 16
MP_L_HIP = 23
MP_R_HIP = 24
MP_L_KNEE = 25
MP_R_KNEE = 26
MP_L_ANKLE = 27
MP_R_ANKLE = 28


# ═══════════════════════════════════════════════════════════════
#  SKELETON CONNECTIONS  (for drawing)
# ═══════════════════════════════════════════════════════════════

# COCO-17 skeleton (YOLOv8-Pose)
COCO_SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # head
    (5, 6),                                    # shoulders
    (5, 7), (7, 9),                            # left arm
    (6, 8), (8, 10),                           # right arm
    (5, 11), (6, 12), (11, 12),                # torso
    (11, 13), (13, 15),                        # left leg
    (12, 14), (14, 16),                        # right leg
]

# MediaPipe-33 skeleton (BlazePose)
MEDIAPIPE_SKELETON_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),           # right face
    (0, 4), (4, 5), (5, 6), (6, 8),           # left face
    (9, 10),                                    # mouth
    (11, 12),                                   # shoulders
    (11, 13), (13, 15),                         # left arm
    (12, 14), (14, 16),                         # right arm
    (11, 23), (12, 24), (23, 24),              # torso
    (23, 25), (25, 27),                         # left leg
    (24, 26), (26, 28),                         # right leg
    (27, 29), (29, 31), (27, 31),              # left foot
    (28, 30), (30, 32), (28, 32),              # right foot
]


# ═══════════════════════════════════════════════════════════════
#  DRAWING DEFAULTS
# ═══════════════════════════════════════════════════════════════

# Colors (BGR)
COLOR_SKELETON = (255, 255, 255)      # white bones
COLOR_KEYPOINT = (0, 255, 255)        # cyan keypoints
COLOR_CONE     = (0, 165, 255)        # orange cone labels
COLOR_BALL     = (0, 255, 0)          # green ball
COLOR_ERROR    = (0, 0, 255)          # red errors
COLOR_GOOD     = (0, 200, 0)          # green for good metrics
COLOR_DEFAULT  = (128, 128, 128)      # grey fallback

# Per-class colors for bounding boxes
CLASS_COLORS = {
    0: (0, 255, 0),      # class 0 — green
    1: (255, 0, 0),      # class 1 — blue
    2: (0, 0, 255),      # class 2 — red
    3: (255, 255, 0),    # class 3 — cyan
    4: (255, 0, 255),    # class 4 — magenta
    5: (0, 255, 255),    # class 5 — yellow
}

# Font / box drawing
BOX_THICKNESS  = 2
FONT_SCALE     = 0.4
FONT_THICKNESS = 1
LABEL_PADDING  = 2


# ═══════════════════════════════════════════════════════════════
#  PHYSICAL DEFAULTS
# ═══════════════════════════════════════════════════════════════

DEFAULT_PLAYER_HEIGHT_M   = 1.75     # assumed player height in metres
DEFAULT_FOOTBALL_DIAMETER_M = 0.215  # 21.5 cm standard size-5 ball
