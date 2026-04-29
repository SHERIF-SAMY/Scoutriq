"""
run_batch.py — Script for batch processing multiple videos in a folder.

Usage:
  Update the BATCH SETTINGS below, then press ▶️ Run.
"""

import os
import sys

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

from config import DrillConfig
from run_drill import get_analyzer_class

# ══════════════════════════════════════════════════════════
#  ⚙️  BATCH SETTINGS
# ══════════════════════════════════════════════════════════

DRILL          = "jumping_15"      # Options: "weakfoot", "diamond", "seven_cone", "jump", "shooting", etc.
INPUT_FOLDER   = r"videos\Football\input\Physical\jump_15"
OUTPUT_DIR     = r"videos\Football\output"

# Model paths (relative to project root)
WEIGHTS = {
    "object_model_path":    r"weights\best2.onnx",
    "pose_model_path":      r"weights\yolov8m-pose.onnx",
    "tracker_config_path":  r"weights\botsort_custom.yaml",
}

POSE_BACKEND    = "yolo"           
PLAYER_HEIGHT   = 1.75             
DEVICE          = "cuda"           
BATCH_SIZE      = 16               # Increase (e.g., 24, 32) to speed up processing if your GPU has enough memory
OBJECT_CONF   = 0.5             # Lower this to detect farther/smaller cones and balls (Default was 0.5)

# Supported video formats
ALLOWED_EXTENSIONS = ('.mp4', '.mov', '.avi')

# ══════════════════════════════════════════════════════════
#  🚀  RUN BATCH
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    _package_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(_package_dir)

    if not os.path.exists(INPUT_FOLDER):
        print(f"[ERROR] Input folder not found: {INPUT_FOLDER}")
        sys.exit(1)

    # 1. Setup config exactly once
    config = DrillConfig(
        object_model_path=WEIGHTS["object_model_path"],
        pose_model_path=WEIGHTS["pose_model_path"],
        tracker_config_path=WEIGHTS["tracker_config_path"],
        pose_backend=POSE_BACKEND,
        player_height_m=PLAYER_HEIGHT,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        object_confidence=OBJECT_CONF,
    )

    # Load the analyzer class
    try:
        AnalyzerClass = get_analyzer_class(DRILL)
    except ValueError as e:
        print(f"[ERROR] Error: {e}")
        sys.exit(1)

    analyzer = AnalyzerClass(config)

    # 2. Find all videos in the incoming folder and subfolders
    video_files = []
    for root, dirs, files in os.walk(INPUT_FOLDER):
        for file in files:
            if file.lower().endswith(ALLOWED_EXTENSIONS):
                full_path = os.path.join(root, file)
                video_files.append(full_path)

    if not video_files:
        print(f"[WARNING] No videos found in {INPUT_FOLDER} with extensions {ALLOWED_EXTENSIONS}")
        sys.exit(0)

    print(f"[INFO] Found {len(video_files)} videos in '{INPUT_FOLDER}'. Starting batch processing...\n")

    # 3. Process each video in a loop
    success_count = 0
    fail_count = 0

    for i, video_path in enumerate(video_files, start=1):
        print(f"\n{'#' * 60}")
        print(f"[RUN] Processing Video {i}/{len(video_files)}: {os.path.basename(video_path)}")
        print(f"{'#' * 60}")

        try:
            report = analyzer.run(video_path, OUTPUT_DIR)
            print(f"\n[SUCCESS] Video {i} completed! Final score: {report.get('overall_score', 'N/A')}/100")
            success_count += 1
        except Exception as e:
            import traceback
            print(f"\n[FAILED] Failed to process video {i}: {os.path.basename(video_path)}")
            print(f"Error details:")
            traceback.print_exc()
            fail_count += 1

    # 4. Final Summary
    print(f"\n{'=' * 60}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"Total Videos : {len(video_files)}")
    print(f"Successful   : {success_count}")
    print(f"Failed       : {fail_count}")
    print(f"{'=' * 60}")
