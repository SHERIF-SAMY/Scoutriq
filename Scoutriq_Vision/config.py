"""
config.py — Configuration dataclasses for ScoutAI drills.

Replaces all hardcoded paths and magic numbers scattered across scripts.
Configs can be constructed in code, loaded from YAML, or overridden via CLI.
"""

from __future__ import annotations

import os
import yaml
from dataclasses import dataclass, field
from typing import Optional

try:
    from .constants import DEFAULT_PLAYER_HEIGHT_M, DEFAULT_FOOTBALL_DIAMETER_M
except ImportError:
    from constants import DEFAULT_PLAYER_HEIGHT_M, DEFAULT_FOOTBALL_DIAMETER_M


@dataclass
class DrillConfig:
    """Configuration for a drill analysis run."""

    # ── Model paths ──
    object_model_path: str = r"weights\best2.onnx"  # YOLO .onnx for cones/ball/goal detection
    pose_model_path: str = ""            # YOLO-Pose or MediaPipe model
    tracker_config_path: str = ""        # BoT-SORT yaml (optional)

    # ── Pose backend ──
    pose_backend: str = "yolo"           # "yolo" or "mediapipe"

    # ── Physical constants ──
    player_height_m: float = DEFAULT_PLAYER_HEIGHT_M
    football_diameter_m: float = DEFAULT_FOOTBALL_DIAMETER_M
    device: str = "cuda"                  # "cpu", "cuda", or "mps"

    # ── Detection thresholds ──
    object_confidence: float = 0.5      # YOLO object detection confidence
    object_imgsz: int = 1024            # Object inference resolution (1920 ensures tiny cones are not lost)
    pose_confidence: float = 0.5         # pose estimation confidence
    pose_iou: float = 0.7               # pose NMS IoU threshold
    keypoint_confidence: float = 0.3     # minimum keypoint visibility
    pose_imgsz: int = 1024              # Inference resolution (increases detection range)

    # ── Smoothing ──
    kp_smoothing_factor: float = 0.3     # keypoint temporal smoothing (0–1)
    kp_history_size: int = 3             # keypoint history frames
    position_smoothing: float = 0.3      # EMA alpha for position

    # ── Tracker ──
    stable_tracker_threshold: float = 80.0   # pixel distance for ID matching
    player_tracker_threshold: float = 150.0  # player re-ID distance

    # ── Drill-specific (overridden per drill) ──
    drill_params: dict = field(default_factory=dict)

    # ── Output ──
    output_dir: str = r"video\output"
    batch_size: int = 16                # Batch size for inference (1 = serial, 4+ = faster on GPU)

    # ── Class name mappings ──
    ball_class_names: list = field(default_factory=lambda: ["football", "ball"])
    cone_class_names: list = field(default_factory=lambda: ["cone"])
    goal_class_names: list = field(default_factory=lambda: ["goal"])

    @classmethod
    def from_yaml(cls, path: str) -> "DrillConfig":
        """Load config from a YAML file, merging with defaults."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def merge(self, overrides: dict) -> "DrillConfig":
        """Return a new config with overrides applied (e.g., from CLI args)."""
        import dataclasses
        current = dataclasses.asdict(self)
        current.update({k: v for k, v in overrides.items() if v is not None})
        return DrillConfig(**current)

    def resolve_paths(self, base_dir: str) -> None:
        """Resolve relative paths against a base directory (in-place)."""
        for attr in ("object_model_path", "pose_model_path", "tracker_config_path"):
            val = getattr(self, attr)
            if val and not os.path.isabs(val):
                setattr(self, attr, os.path.join(base_dir, val))
