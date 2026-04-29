import os
from dotenv import load_dotenv

load_dotenv()

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ── GPU startup banner ──
try:
    import torch
    from Scoutriq_Vision.core.gpu_utils import get_device_label, configure_torch, get_device
    _device = get_device()
    configure_torch(_device)
    print(get_device_label())
except ImportError:
    print("Scoutriq Vision: torch not installed — running in CPU-only mode")

import shutil
import tempfile
import time
import glob
import subprocess
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# Ensure output directory exists at startup
os.makedirs("output", exist_ok=True)

def reencode_video_h264(input_path: str) -> str:
    """Re-encode a video to H.264 codec for browser playback."""
    output_path = input_path.replace(".mp4", "_h264.mp4")
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-c:v", "libx264", "-preset", "fast",
            "-crf", "23", "-movflags", "+faststart",
            "-pix_fmt", "yuv420p",
            output_path
        ], check=True, capture_output=True, timeout=120)
        print(f"Re-encoded video to H.264: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.decode("utf-8", errors="replace")
        print(f"Warning: ffmpeg re-encode failed with libx264:\n{err_msg}")
        
        # Fallback to MediaFoundation h264 for Windows environments (conda default ffmpeg missing libx264)
        try:
            print("Attempting Windows fallback H.264 codec (h264_mf)...")
            subprocess.run([
                "ffmpeg", "-y", "-i", input_path,
                "-c:v", "h264_mf",
                "-b:v", "3M",
                "-movflags", "+faststart",
                output_path
            ], check=True, capture_output=True, timeout=120)
            print(f"Re-encoded video to H.264 using h264_mf: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e2:
            print(f"Warning: Fallback ffmpeg re-encode failed:\n{e2.stderr.decode('utf-8', errors='replace')}")
            return input_path
        except Exception as e2:
            print(f"Warning: Fallback ffmpeg execution failed: {e2}")
            return input_path
    except Exception as e:
        print(f"Warning: ffmpeg execution failed: {e}")
        return input_path


def normalize_video_orientation(input_path: str) -> str:
    """
    Bake rotation metadata into pixel data so OpenCV reads frames correctly.
    Fixes upside-down / sideways videos recorded on iPhone / Android (.mov / .mp4).
    ffmpeg respects the 'rotate' metadata by default; OpenCV ignores it.
    Returns the path to a normalized .mp4, or the original path if conversion fails.
    """
    output_path = os.path.splitext(input_path)[0] + "_oriented.mp4"
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-c:v", "libx264", "-preset", "fast",
            "-crf", "18", "-movflags", "+faststart",
            "-pix_fmt", "yuv420p",
            output_path
        ], check=True, capture_output=True, timeout=300)
        print(f"[Orientation] Normalized video: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode("utf-8", errors="replace")
        print(f"[Orientation] libx264 failed, trying h264_mf fallback:\n{err[:300]}")
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", input_path,
                "-c:v", "h264_mf", "-b:v", "5M",
                "-movflags", "+faststart",
                output_path
            ], check=True, capture_output=True, timeout=300)
            print(f"[Orientation] Normalized via h264_mf: {output_path}")
            return output_path
        except Exception as e2:
            print(f"[Orientation] Fallback also failed: {e2} — using original")
            return input_path
    except Exception as e:
        print(f"[Orientation] ffmpeg not available: {e} — using original")
        return input_path


def convert_numpy(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
from fastapi.middleware.cors import CORSMiddleware

# Import from Scoutriq Vision
from Scoutriq_Vision.config import DrillConfig
from Scoutriq_Vision.run_drill import get_analyzer_class, DRILL_REGISTRY
# [LLM DISABLED] from Scoutriq_Vision.core.llm_feedback import generate_llm_feedback


# ─── Response Simplifiers (dashboard-friendly JSON) ───

def _simplify_weakfoot(report: dict) -> dict:
    speed = report.get("speed_metrics", {})
    shot = report.get("shot_metrics", {})
    balance = report.get("balance_metrics", {})
    dribble = report.get("dribbling_metrics", {})
    foot = report.get("foot_usage", {})
    return {
        "dribblingMetrics": {
            "controlPercentage": dribble.get("control_percentage", 0),
            "ballPossessionPercentage": dribble.get("ball_possession_percentage", 0),
            "averageBallDistanceM": dribble.get("average_ball_distance_m", 0),
            "touchCount": dribble.get("touch_count", 0),
        },
        "speedMetrics": {
            "averageSpeed": speed.get("average_speed_m_per_s", 0),
            "maxSpeed": speed.get("max_speed_m_per_s", 0),
            "dribbleAvgSpeed": speed.get("dribble_phase_avg_speed_m_per_s", 0),
        },
        "shotMetrics": {
            "shotDetected": shot.get("shot_detected", False),
            "shotPowerScore": shot.get("shot_power_score"),
            "shotTechniqueScore": shot.get("shot_technique_score"),
            "maxBallSpeedMps": shot.get("max_ball_speed_m_per_s"),
            "feedback": shot.get("feedback", []),
        },
        "balanceMetrics": {
            "balanceScore": balance.get("balance_score", 0),
        },
        "footUsage": {
            "weakFoot": foot.get("weak_foot", report.get("drill_info", {}).get("weak_foot")),
            "weakFootUsagePercentage": foot.get("weak_foot_usage_percentage", 0),
        },
    }


def _simplify_seven_cone(report: dict) -> dict:
    time_m = report.get("time_metrics", {})
    touch = report.get("touch_metrics", {})
    ball = report.get("ball_control", {})
    errors = report.get("errors", {})
    return {
        "timeMetrics": {
            "totalDurationSeconds": time_m.get("total_duration_seconds", 0),
        },
        "touchMetrics": {
            "totalTouches": touch.get("total_touches", 0),
            "leftFootTouches": touch.get("left_foot_touches", 0),
            "rightFootTouches": touch.get("right_foot_touches", 0),
            "touchesPerSecond": touch.get("touches_per_second", 0),
        },
        "ballControl": {
            "averageSeparationMeters": ball.get("average_separation_meters", 0),
            "closeControlPercentage": ball.get("close_control_percentage", 0),
            "ballControlScore": ball.get("ball_control_score", 0),
        },
        "errors": {
            "coneContacts": errors.get("cone_contacts", 0),
            "missedCones": errors.get("missed_cones", 0),
            "lossOfControlCount": errors.get("loss_of_control_count", 0),
        },
    }


def _simplify_diamond(report: dict) -> dict:
    time_m = report.get("time_metrics", {})
    speed = report.get("speed_metrics", {})
    ball = report.get("ball_control", {})
    movement = report.get("movement_quality", {})
    pace = report.get("pace_consistency", {})
    return {
        "timeMetrics": {
            "totalDurationSeconds": time_m.get("total_duration_seconds", 0),
        },
        "speedMetrics": {
            "averageSpeedMps": speed.get("average_speed_mps", 0),
            "averageSpeedKmh": speed.get("average_speed_kmh", 0),
            "maxSpeedMps": speed.get("max_speed_mps", 0),
            "maxSpeedKmh": speed.get("max_speed_kmh", 0),
            "totalDistanceMeters": speed.get("total_distance_meters", 0),
        },
        "ballControl": {
            "averageDistanceMeters": ball.get("average_distance_meters", 0),
            "possessionPercentage": ball.get("possession_percentage", 0),
            "ballControlScore": ball.get("ball_control_score", 0),
        },
        "movementQuality": {
            "balanceScore": movement.get("balance_score", 0),
            "agilityScore": movement.get("agility_score", 0),
        },
        "paceConsistency": {
            "consistencyScore": pace.get("consistency_score", 0),
            "feedback": pace.get("feedback", []),
        },
    }


def _simplify_jump(report: dict) -> dict:
    time_m = report.get("time_metrics", {})
    jump = report.get("jump_metrics", {})
    knee = report.get("knee_angle_metrics", {})
    # Compute an overall score for jump (raw report doesn't have one)
    max_h = jump.get("max_jump_height_cm", 0)
    bent = knee.get("knee_bent_during_jump", False)
    bend_count = knee.get("knee_bend_events_count", 0)
    height_score = min(100, max_h * 2)  # 50cm jump = 100
    form_penalty = bend_count * 5 if bent else 0
    overall = max(0, min(100, height_score - form_penalty))
    return {
        "timeMetrics": {
            "totalDurationSeconds": time_m.get("total_duration_seconds", 0),
        },
        "jumpMetrics": {
            "maxJumpHeightCm": jump.get("max_jump_height_cm", 0),
        },
        "kneeAngleMetrics": {
            "kneeBentDuringJump": bent,
            "kneeBendEventsCount": bend_count,
        },
        "overallScore": round(overall, 1),
    }


def _simplify_shooting(report: dict) -> dict:
    sm = report.get("shooting_metrics", {})
    shots = report.get("shots", [])
    # Simplify each shot
    simplified_shots = []
    for s in shots:
        simplified_shots.append({
            "shotNumber": s.get("shot_number"),
            "foot": s.get("foot"),
            "insideGate": s.get("inside_gate"),
            "goal": s.get("goal"),
            "goalZoneValue": s.get("goal_zone_value"),
            "goalZoneLabel": s.get("goal_zone_label"),
            "shotSpeedMs": s.get("shot_speed_ms"),
            "missDistanceM": s.get("miss_distance_m"),
        })
    return {
        "shootingMetrics": {
            "totalTimeS": sm.get("total_time_s", 0),
            "totalTouches": sm.get("total_touches", 0),
            "totalShots": sm.get("total_shots", 0),
            "goalsScored": sm.get("goals_scored", 0),
            "shotsMissed": sm.get("shots_missed", 0),
            "accuracyPct": sm.get("accuracy_pct", 0),
            "shotsInsideGate": sm.get("shots_inside_gate", 0),
            "shotsOutsideGate": sm.get("shots_outside_gate", 0),
            "avgGoalZoneValue": sm.get("avg_goal_zone_value", 0),
        },
        "shots": simplified_shots,
        "errors": report.get("errors", []),
    }


def _simplify_t_test(report: dict) -> dict:
    sections = report.get("sections", {})
    kinematics = report.get("overall_kinematics", {})
    cod = report.get("overall_change_of_direction", {})
    dwell = report.get("change_of_direction_dwell_times", {})
    scoring = report.get("scoring", {})
    info = report.get("drill_info", {})
    return {
        "drillCompleted": info.get("completed", False),
        "totalTimeS": info.get("total_time_s", 0),
        "sections": sections,
        "overallKinematics": {
            "maxSectionSpeedMs": kinematics.get("max_section_speed_m_s", 0),
        },
        "changeOfDirection": {
            "avgDwellTimeS": cod.get("avg_dwell_time_s", 0),
            "totalCodEvents": cod.get("total_cod_events", 0),
            "dwellTimesPerCone": dwell,
        },
        "errors": report.get("errors", {}),
        "scoring": {
            "timeScore": scoring.get("time_score", 0),
            "errorPenalty": scoring.get("error_penalty", 0),
            "formPenalty": scoring.get("form_penalty", 0),
            "overallScore": scoring.get("overall_score", 0),
        },
    }


def _simplify_jumping_15(report: dict) -> dict:
    info = report.get("drill_info", {})
    jump = report.get("jump_metrics", {})
    pace = report.get("pace_consistency", {})
    form = report.get("form_assessment", {})
    return {
        "timeAnalyzedS": info.get("time_analyzed_s", 0),
        "jumpMetrics": {
            "totalJumps": jump.get("total_jumps", 0),
            "volumeScore": jump.get("volume_score", 0),
        },
        "paceConsistency": {
            "avgTimePerJumpS": pace.get("average_time_per_jump_s", 0),
            "stdDevS": pace.get("std_dev_s", 0),
            "coefficientOfVariationPct": pace.get("coefficient_of_variation_percent", 0),
            "consistencyScore": pace.get("consistency_score", 0),
            "slowJumpsCount": pace.get("slow_jumps_count", 0),
            "fastJumpsCount": pace.get("fast_jumps_count", 0),
            "feedback": pace.get("feedback", []),
        },
        "formAssessment": {
            "symmetryScore": form.get("symmetry_score", 0),
            "asymmetricJumpsCount": form.get("asymmetric_jumps_count", 0),
            "feedback": form.get("feedback", ""),
        },
    }


_DRILL_SIMPLIFIERS = {
    "weakfoot": _simplify_weakfoot,
    "seven_cone": _simplify_seven_cone,
    "diamond": _simplify_diamond,
    "jump": _simplify_jump,
    "shooting": _simplify_shooting,
    "t_test": _simplify_t_test,
    "jumping_15": _simplify_jumping_15,
}


def simplify_report(drill: str, report: dict) -> dict:
    """Transform the raw analysis report into a dashboard-friendly JSON."""
    simplifier = _DRILL_SIMPLIFIERS.get(drill)
    if simplifier is None:
        return report  # unknown drill — return raw

    simplified = {
        "playerId": report.get("player_id"),
        "drillType": drill,
        "video": {
            "analyzedVideoUrl": report.get("compressed_video_url"),  # H.264 — browser-playable with overlays
            "rawVideoUrl": report.get("raw_video_url"),               # H.264 — same content, separate URL
        },
    }
    simplified.update(simplifier(report))

    # Common fields
    if "overall_score" in report and "overallScore" not in simplified:
        simplified["overallScore"] = report["overall_score"]
    simplified["formAssessment"] = report.get("form_assessment", "")
    # [LLM DISABLED] simplified["llmFeedback"] = report.get("llm_feedback", "")

    return simplified


app = FastAPI(
    title="ScoutAI Video Analysis API",
    description="API for analyzing sports drills from video files.",
    version="1.0.0"
)

# Enable CORS for frontend integration — MUST be added before mounting static files
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the output directory to serve analyzed videos directly
app.mount("/videos", StaticFiles(directory="output"), name="videos")
# Frontend is served separately — not included in Docker image


@app.get("/")
def read_root():
    return {"message": "Welcome to the ScoutAI API. Use /docs to see available endpoints."}

@app.get("/drills")
def list_drills():
    """List all available drills and their descriptions."""
    return [{"name": name, "description": info["description"]} for name, info in DRILL_REGISTRY.items()]


@app.post("/analyze-drill")
async def analyze_drill(
    request: Request,
    video: UploadFile = File(...),
    drill: str = Form(...),
    player_id: str = Form(...),
    pose_backend: str = Form("yolo"),
    player_height: float = Form(1.75)
):
    """
    Analyze a video file for a specific drill.
    """
    if drill not in DRILL_REGISTRY:
        raise HTTPException(
            status_code=400, 
            detail=f"Unknown drill: '{drill}'. Available drills: {', '.join(DRILL_REGISTRY.keys())}"
        )

    # Ensure output directory exists — organized by player_id
    output_dir = os.path.join(os.getcwd(), "output", player_id)
    os.makedirs(output_dir, exist_ok=True)

    # Save uploaded video to a temporary file
    try:
        # Create a temp file path
        fd, temp_video_path = tempfile.mkstemp(suffix=os.path.splitext(video.filename)[1])
        os.close(fd)
        
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
            
        print(f"Video saved to {temp_video_path}")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save video: {str(e)}")
    
    finally:
        video.file.close()

    # Run Analysis
    try:
        print(f"Starting analysis for drill: {drill}")
        configs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Scoutriq_Vision", "configs")

        drill_config_path = os.path.join(configs_dir, f"{drill}.yaml")
        default_config_path = os.path.join(configs_dir, "default.yaml")

        # Always load default config first as the base
        if os.path.exists(default_config_path):
            print(f"Loading default config: {default_config_path}")
            config = DrillConfig.from_yaml(default_config_path)
        else:
            config = DrillConfig()

        # Merge drill-specific config on top if it exists
        if os.path.exists(drill_config_path):
            print(f"Loading drill-specific config overrides: {drill_config_path}")
            import yaml
            with open(drill_config_path, "r", encoding="utf-8") as f:
                drill_overrides = yaml.safe_load(f) or {}
            config = config.merge(drill_overrides)

        # Always apply user-provided overrides (pose_backend, player_height, output_dir)
        config = config.merge({
            "pose_backend": pose_backend,
            "player_height_m": player_height,
            "output_dir": output_dir,
        })
        
        AnalyzerClass = get_analyzer_class(drill)
        analyzer = AnalyzerClass(config)

        # Normalize video orientation BEFORE analysis
        # ffmpeg bakes rotation metadata into pixels; OpenCV ignores that metadata
        # This fixes upside-down .mov files from iPhone/iPad cameras
        print(f"[Orientation] Normalizing video orientation...")
        oriented_video_path = normalize_video_orientation(temp_video_path)

        # Run the tracking/analysis on the orientation-corrected video
        report = analyzer.run(oriented_video_path, output_dir)

        # Clean up orientation-normalized temp file if it was created
        if oriented_video_path != temp_video_path and os.path.exists(oriented_video_path):
            try:
                os.remove(oriented_video_path)
            except Exception:
                pass
        
        # [LLM DISABLED] — uncomment to re-enable AI coaching feedback
        # print(f"Generating LLM feedback for drill: {drill}")
        # feedback = generate_llm_feedback(drill, report)
        # report["llm_feedback"] = feedback
        
        # Build direct URLs — search recursively in output/<player_id>/
        all_mp4s = glob.glob(os.path.join(output_dir, "**", "*.mp4"), recursive=True)
        # Exclude any re-encoded variants so we always pick the freshly analyzed video
        video_files = [
            f for f in all_mp4s
            if not f.endswith("_h264.mp4")
            and not f.endswith("_compressed.mp4")
            and not f.endswith("_raw_h264.mp4")
        ]
        if video_files:
            original_video = max(video_files, key=os.path.getmtime)

            # Re-encode to H.264 for analyzedVideoUrl (browser-playable)
            compressed_video = reencode_video_h264(original_video)

            # Re-encode again to a separate file for rawVideoUrl
            raw_h264_path = original_video.replace(".mp4", "_raw_h264.mp4")
            if compressed_video != original_video and os.path.exists(compressed_video):
                # Copy the H.264 version instead of re-encoding a second time
                shutil.copy2(compressed_video, raw_h264_path)
                raw_video = raw_h264_path
            else:
                raw_video = original_video  # fallback

            output_root = os.path.join(os.getcwd(), "output")
            base_url = str(request.base_url).rstrip("/")

            comp_rel = os.path.relpath(compressed_video, output_root).replace("\\", "/")
            raw_rel  = os.path.relpath(raw_video, output_root).replace("\\", "/")

            report["compressed_video_url"] = f"{base_url}/videos/{comp_rel}"  # H.264 analyzed
            report["raw_video_url"]         = f"{base_url}/videos/{raw_rel}"   # H.264 raw copy
            print(f"analyzedVideoUrl: {report['compressed_video_url']}")
            print(f"rawVideoUrl     : {report['raw_video_url']}")
        else:
            print("No output video found in directory.")
            report["compressed_video_url"] = None
            report["raw_video_url"]         = None
        
        report["player_id"] = player_id
        simplified = simplify_report(drill, convert_numpy(report))
        return JSONResponse(status_code=200, content=simplified)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
        
    finally:
        # Clean up temporary video file
        if os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except Exception as e:
                print(f"Warning: Failed to remove temporary file {temp_video_path}: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
