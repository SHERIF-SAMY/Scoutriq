"""
overlay.py — Metrics overlay panel for video output.

Provides a reusable MetricsPanel class that drills can populate
with their specific metrics.  The panel draws a semi-transparent
dark background with color-coded text rows.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional

from ..constants import COLOR_GOOD, COLOR_ERROR


class MetricsPanel:
    """
    A reusable semi-transparent overlay panel for displaying live metrics.

    Usage:
        panel = MetricsPanel()
        panel.add("Time", "12.3s")
        panel.add("Touches", "15  (L:8 R:7)")
        panel.add("Ball Dist", "0.42m", color=COLOR_GOOD)
        panel.draw(frame)
    """

    def __init__(
        self,
        x: int = 5,
        y: int = 5,
        width: int = 380,
        line_height: int = 22,
        font_scale: float = 0.48,
        opacity: float = 0.65,
    ):
        self.x = x
        self.y = y
        self.width = width
        self.line_height = line_height
        self.font_scale = font_scale
        self.opacity = opacity
        self._rows: list[tuple[str, tuple[int, int, int]]] = []

    def clear(self) -> None:
        """Clear all rows (call before each frame)."""
        self._rows.clear()

    def add(
        self,
        text: str,
        color: tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        """Add a row of text with optional color."""
        self._rows.append((text, color))

    def add_metric(
        self,
        label: str,
        value: str,
        good: bool = True,
    ) -> None:
        """Add a metric row, auto-colored green/red based on 'good' flag."""
        color = COLOR_GOOD if good else COLOR_ERROR
        self.add(f"{label}: {value}", color)

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Draw the panel onto the frame and return it."""
        if not self._rows:
            return frame

        panel_h = len(self._rows) * self.line_height + 15
        h, w = frame.shape[:2]

        # Clamp ROI to frame bounds
        y1 = max(0, self.y)
        y2 = min(h, self.y + panel_h)
        x1 = max(0, self.x)
        x2 = min(w, self.x + self.width)

        if y2 > y1 and x2 > x1:
            roi = frame[y1:y2, x1:x2].copy()       # Small copy (~30KB vs ~6MB)
            cv2.rectangle(roi, (0, 0), (x2 - x1, y2 - y1), (0, 0, 0), -1)
            cv2.addWeighted(
                roi, self.opacity,
                frame[y1:y2, x1:x2], 1 - self.opacity,
                0, frame[y1:y2, x1:x2],
            )

        text_y = self.y + 20
        for text, color in self._rows:
            cv2.putText(
                frame,
                text,
                (self.x + 7, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                color,
                1,
                cv2.LINE_AA,
            )
            text_y += self.line_height

        return frame


def draw_error_banner(
    frame: np.ndarray,
    errors: list[str],
    color: tuple[int, int, int] = (0, 0, 180),
) -> np.ndarray:
    """
    Flash error banners at the bottom-center of the frame.

    Args:
        frame: BGR image.
        errors: List of error strings to display.
        color: Background color for error banners.

    Returns:
        Annotated frame.
    """
    if not errors:
        return frame

    h, w = frame.shape[:2]
    unique = list(set(errors))
    for i, err in enumerate(unique):
        text = f"!! {err}"
        (tw, th), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
        )
        ex = w // 2 - tw // 2
        ey = h - 40 - i * (th + 15)
        cv2.rectangle(
            frame,
            (ex - 5, ey - th - 5),
            (ex + tw + 5, ey + 5),
            color,
            -1,
        )
        cv2.putText(
            frame,
            text,
            (ex, ey),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return frame
