"""
geometry.py — Pure geometry utilities.

Extracted from functions duplicated across diamond.py, seven_cone_dribble.py,
weakfoot_ultimate.py, etc.  All functions are stateless.
"""

from __future__ import annotations

import math
import numpy as np


def dist2d(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Euclidean distance between two 2D points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def box_center(box) -> tuple[float, float]:
    """Centre of an (x1, y1, x2, y2) bounding box."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def box_size(box) -> tuple[float, float]:
    """(width, height) of an (x1, y1, x2, y2) bounding box."""
    x1, y1, x2, y2 = box
    return (x2 - x1, y2 - y1)


def angle_between(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> float:
    """
    Angle at point *b* formed by segments ba and bc, in degrees.

    Used for knee angles, trunk lean, etc.
    """
    ba = np.array([a[0] - b[0], a[1] - b[1]], dtype=np.float64)
    bc = np.array([c[0] - b[0], c[1] - b[1]], dtype=np.float64)

    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def iou(box_a, box_b) -> float:
    """Intersection-over-Union of two (x1, y1, x2, y2) boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
