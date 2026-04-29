"""
run_drill.py — CLI entry point for ScoutAI drill analysis.

Usage:
    python run_drill.py --drill seven_cone --video path/to/video.mp4 --output ./output
    python run_drill.py --drill diamond --video video.mp4 --config configs/diamond.yaml
    python run_drill.py --list                     # list available drills
"""

from __future__ import annotations

import argparse
import os
import sys

from .config import DrillConfig


# ── Drill registry ──
DRILL_REGISTRY = {
    "seven_cone": {
        "module": "Scoutriq_Vision.drills.seven_cone",
        "class": "SevenConeDrillAnalyzer",
        "description": "7-cone dribble drill (line of cones, dribble + turn)",
    },
    "diamond": {
        "module": "Scoutriq_Vision.drills.diamond",
        "class": "DiamondDrillAnalyzer",
        "description": "Diamond cone sprint drill (4 cones, agility test)",
    },
    "weakfoot": {
        "module": "Scoutriq_Vision.drills.weakfoot",
        "class": "WeakFootAnalyzer",
        "description": "Weak foot dribble & shoot drill",
    },
    "jump": {
        "module": "Scoutriq_Vision.drills.jump",
        "class": "JumpAnalyzer",
        "description": "Vertical / countermovement jump test",
    },
    "jumping_15": {
        "module": "Scoutriq_Vision.drills.jumping_15",
        "class": "Jumping15Analyzer",
        "description": "Side-to-side jumps over ball for 15s",
    },
    "t_test": {
        "module": "Scoutriq_Vision.drills.t_test",
        "class": "TTestAnalyzer",
        "description": "T-Test Agility Drill",
    },
    "shooting": {
        "module": "Scoutriq_Vision.drills.shooting",
        "class": "ShootingDrillAnalyzer",
        "description": "Shooting drill (3 shots through cone gate at goal)",
    },
}


def get_analyzer_class(drill_name: str):
    """Dynamically import and return the analyzer class."""
    info = DRILL_REGISTRY.get(drill_name)
    if info is None:
        available = ", ".join(DRILL_REGISTRY.keys())
        raise ValueError(
            f"Unknown drill: '{drill_name}'. Available: {available}"
        )

    import importlib
    module = importlib.import_module(info["module"])
    return getattr(module, info["class"])


def main():
    parser = argparse.ArgumentParser(
        description="ScoutAI — Sports Drill Video Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_drill.py --drill seven_cone --video "video.MOV" --output ./output
  python run_drill.py --drill diamond --video "video.mp4" --config configs/diamond.yaml
  python run_drill.py --list
        """,
    )

    parser.add_argument(
        "--drill", type=str, default=None,
        help=f"Drill to analyze ({', '.join(DRILL_REGISTRY.keys())})",
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Path to input video file",
    )
    parser.add_argument(
        "--output", type=str, default="./output",
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--object-model", type=str, default=None,
        help="Path to YOLO object detection model (.onnx)",
    )
    parser.add_argument(
        "--pose-model", type=str, default=None,
        help="Path to pose estimation model",
    )
    parser.add_argument(
        "--pose-backend", type=str, default=None,
        choices=["yolo", "mediapipe"],
        help="Pose estimation backend (default: yolo)",
    )
    parser.add_argument(
        "--tracker-config", type=str, default=None,
        help="Path to BoT-SORT tracker config YAML",
    )
    parser.add_argument(
        "--player-height", type=float, default=None,
        help="Player height in metres (default: 1.75)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available drills and exit",
    )

    args = parser.parse_args()

    # ── List drills ──
    if args.list:
        print("\nAvailable drills:")
        print("-" * 50)
        for name, info in DRILL_REGISTRY.items():
            print(f"  {name:15s}  {info['description']}")
        print()
        return

    # ── Validate args ──
    if args.drill is None:
        parser.error("--drill is required. Use --list to see available drills.")
    if args.video is None:
        parser.error("--video is required.")

    # ── Load config ──
    if args.config:
        config = DrillConfig.from_yaml(args.config)
    else:
        config = DrillConfig()

    # ── Apply CLI overrides ──
    overrides = {}
    if args.object_model:
        overrides["object_model_path"] = args.object_model
    if args.pose_model:
        overrides["pose_model_path"] = args.pose_model
    if args.pose_backend:
        overrides["pose_backend"] = args.pose_backend
    if args.tracker_config:
        overrides["tracker_config_path"] = args.tracker_config
    if args.player_height:
        overrides["player_height_m"] = args.player_height
    if args.output:
        overrides["output_dir"] = args.output

    if overrides:
        config = config.merge(overrides)

    # ── Resolve relative paths (only for YAML-loaded configs) ──
    if args.config:
        config_dir = os.path.dirname(os.path.abspath(args.config))
        config.resolve_paths(config_dir)

    # ── Create and run analyzer ──
    AnalyzerClass = get_analyzer_class(args.drill)
    analyzer = AnalyzerClass(config)
    report = analyzer.run(args.video, config.output_dir)

    # Print summary
    print(f"\n{'=' * 55}")
    print(f"  ⭐ OVERALL SCORE: {report.get('overall_score', 'N/A')}/100")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
