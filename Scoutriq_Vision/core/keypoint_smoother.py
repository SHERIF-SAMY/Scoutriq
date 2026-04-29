"""
keypoint_smoother.py — Adaptive temporal smoothing for pose keypoints
                       using the One Euro Filter algorithm.

The One Euro Filter is the industry standard for real-time signal
smoothing in motion capture and sports CV.  It is speed-adaptive:

  • When a keypoint moves slowly  → heavy smoothing (removes jitter)
  • When a keypoint moves quickly → light smoothing (prevents lag)

This replaces the previous fixed-weight EMA smoother.

Reference:
    Casiez, Roussel & Vogel (2012)
    "1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input"
    https://cristal.univ-lille.fr/~casiez/1euro/
"""

from __future__ import annotations

import math
import numpy as np
from collections import defaultdict


# ═══════════════════════════════════════════════════════════
#  Low-level One Euro Filter (operates on a single scalar)
# ═══════════════════════════════════════════════════════════

class _LowPassFilter:
    """Simple first-order low-pass (exponential) filter."""

    __slots__ = ("_y", "_initialized")

    def __init__(self):
        self._y: float = 0.0
        self._initialized: bool = False

    def __call__(self, value: float, alpha: float) -> float:
        if not self._initialized:
            self._y = value
            self._initialized = True
        else:
            self._y = alpha * value + (1.0 - alpha) * self._y
        return self._y

    @property
    def last(self) -> float:
        return self._y

    def reset(self):
        self._initialized = False
        self._y = 0.0


class _OneEuroScalar:
    """One Euro Filter for a single scalar value."""

    __slots__ = ("_min_cutoff", "_beta", "_d_cutoff", "_x_filter", "_dx_filter")

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self._x_filter = _LowPassFilter()
        self._dx_filter = _LowPassFilter()

    @staticmethod
    def _alpha(cutoff: float, te: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: float, te: float = 1.0) -> float:
        # Estimate derivative
        prev = self._x_filter.last if self._x_filter._initialized else x
        dx = (x - prev) / te
        edx = self._dx_filter(dx, self._alpha(self._d_cutoff, te))

        # Adaptive cutoff
        cutoff = self._min_cutoff + self._beta * abs(edx)
        return self._x_filter(x, self._alpha(cutoff, te))

    def reset(self):
        self._x_filter.reset()
        self._dx_filter.reset()


# ═══════════════════════════════════════════════════════════
#  Public API — drop-in replacement for the old smoother
# ═══════════════════════════════════════════════════════════

class KeypointSmoother:
    """
    Adaptive temporal smoother for per-person keypoint arrays
    using the One Euro Filter.

    API is identical to the previous EMA-based smoother so no
    changes are needed in base_drill.py or drill files.

    Args:
        smoothing_factor: Maps to One Euro ``min_cutoff``.
                          Lower = more smoothing when still.
                          Recommended range: 0.05 – 1.0.
                          (Previously was the EMA blend weight.)
        history_size:     Maps to One Euro ``beta``.
                          Higher = less lag during fast movement.
                          Recommended range: 0.001 – 0.05.
                          (Previously was the history buffer size;
                           we reinterpret it as beta × 1000 for
                           backward-compatible config values.)
    """

    def __init__(self, smoothing_factor: float = 0.3, history_size: int = 3):
        # Map old config params to One Euro params
        # smoothing_factor 0.3 → min_cutoff 0.3  (lower = smoother)
        # history_size 3      → beta 0.003       (higher = less lag)
        self._min_cutoff = max(0.01, smoothing_factor)
        self._beta = max(0.001, history_size / 1000.0)
        self._d_cutoff = 1.0

        # Per-person, per-keypoint, per-axis filters
        # filters[person_id] = np.ndarray of _OneEuroScalar (shape N×2 for x,y)
        self._filters: dict[int, list[list[_OneEuroScalar]]] = defaultdict(list)
        self.last_valid: dict[int, np.ndarray] = {}

    def _ensure_filters(self, person_id: int, num_kps: int) -> list[list[_OneEuroScalar]]:
        """Lazily create filters for a person the first time they appear."""
        if not self._filters[person_id]:
            self._filters[person_id] = [
                [
                    _OneEuroScalar(self._min_cutoff, self._beta, self._d_cutoff),
                    _OneEuroScalar(self._min_cutoff, self._beta, self._d_cutoff),
                ]
                for _ in range(num_kps)
            ]
        return self._filters[person_id]

    def smooth(self, person_id: int, keypoints: np.ndarray) -> np.ndarray:
        """
        Smooth a keypoint array for the given person ID.

        Args:
            person_id: Stable tracking ID for this person.
            keypoints: Shape (N, 3) array — [x, y, confidence] per keypoint.

        Returns:
            Smoothed keypoints, same shape.
        """
        keypoints = np.array(keypoints, dtype=np.float32)
        num_kps = len(keypoints)
        filters = self._ensure_filters(person_id, num_kps)

        smoothed = keypoints.copy()
        for i in range(min(num_kps, len(filters))):
            conf = keypoints[i, 2]
            if conf > 0.1:  # Only smooth visible keypoints
                smoothed[i, 0] = filters[i][0](float(keypoints[i, 0]))
                smoothed[i, 1] = filters[i][1](float(keypoints[i, 1]))
            else:
                # Low confidence — reset filter for this keypoint
                # so it doesn't drag toward stale position
                filters[i][0].reset()
                filters[i][1].reset()

        self.last_valid[person_id] = smoothed.copy()
        return smoothed

    def reset(self, person_id: int | None = None) -> None:
        """Clear filters for one person, or all if person_id is None."""
        if person_id is None:
            self._filters.clear()
            self.last_valid.clear()
        else:
            self._filters.pop(person_id, None)
            self.last_valid.pop(person_id, None)
