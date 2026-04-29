"""
ScoutAI — Entry Point.

Two modes:
  1. CLI:       python -m scoutai --drill diamond --video path/to/video.mp4
  2. Direct:    Press ▶️ Run on this file (uses SETTINGS below)
"""

import sys
import os

# ── Ensure imports work when running this file directly ──
_package_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_package_dir)

if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

# Handle case-sensitivity issues: map 'scoutai' to 'ScoutAI'
try:
    import ScoutAI as scoutai_pkg
    sys.modules['scoutai'] = scoutai_pkg
except ImportError:
    pass


# ══════════════════════════════════════════════════════════
#  ⚙️  SETTINGS — Used when pressing ▶️ Run (no CLI args)
# ══════════════════════════════════════════════════════════

DRILL       = "diamond"      # Options: "weakfoot", "diamond", "seven_cone", "jump", "jumping_15", "t_test", "shooting"
VIDEO       = r"videos\Football\input\Technical\Diamond\1_diamond.mp4"
OUTPUT_DIR  = r"videos\Football\output"

# Model paths (relative to project root)
WEIGHTS = {
    "object_model_path":    r"weights\best2.onnx",
    "pose_model_path":      r"weights\yolov8m-pose.onnx", # Ensure this file is in the weights/ folder
    "tracker_config_path":  r"weights\botsort_custom.yaml",
}

POSE_BACKEND    = "yolo"           # "yolo" or "mediapipe"
PLAYER_HEIGHT   = 1.75             # metres
DEVICE          = "cuda"           # "cpu" or "cuda" (Use "cpu" if you get CUDA errors)
BATCH_SIZE      = 16               # Increase (e.g., 24, 32) to speed up processing if your GPU has enough memory
OBJECT_CONF   = 0.5             # Lower this to detect farther/smaller cones and balls (Default was 0.5)
OBJECT_IMGSZ  = 1024            # Object detection inference resolution (Default was 1920)

# ══════════════════════════════════════════════════════════
#  🚀  RUN
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.chdir(_package_dir)  # ensure paths resolve from the ScoutAI folder

    # If CLI args provided → use run_drill CLI
    # If no args → use SETTINGS above (direct run)
    has_cli_args = len(sys.argv) > 1

    if has_cli_args:
        from run_drill import main
        main()
    else:
        from config import DrillConfig
        from run_drill import get_analyzer_class

        config = DrillConfig(
            object_model_path=WEIGHTS["object_model_path"],
            pose_model_path=WEIGHTS["pose_model_path"],
            tracker_config_path=WEIGHTS["tracker_config_path"],
            pose_backend=POSE_BACKEND,
            player_height_m=PLAYER_HEIGHT,
            device=DEVICE,
            batch_size=BATCH_SIZE,
            object_confidence=OBJECT_CONF,
            object_imgsz=OBJECT_IMGSZ,
        )

        AnalyzerClass = get_analyzer_class(DRILL)
        analyzer = AnalyzerClass(config)
        report = analyzer.run(VIDEO, OUTPUT_DIR)

        print(f"\n{'=' * 55}")
        print(f"  ⭐ OVERALL SCORE: {report.get('overall_score', 'N/A')}/100")
        print(f"{'=' * 55}")
