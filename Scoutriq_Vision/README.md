# ScoutAI ⚽️

AI-powered football drill analysis system using real-time pose estimation and object detection to generate performance reports from monocular video.

---

## 🏗️ Architecture Overview

ScoutAI uses a **Template Method** design pattern. The base class (`base_drill.py`) owns the entire video processing pipeline — **subclasses only implement drill-specific hooks**, never the loop itself.

```
┌─────────────────────────────────────────────────────────┐
│  __main__.py / run_batch.py / run_drill.py (CLI)        │  ← Entry points
│  Creates DrillConfig + picks AnalyzerClass              │
├─────────────────────────────────────────────────────────┤
│  BaseDrillAnalyzer (base_drill.py)                      │  ← Template Method
│  run() → load → batch detect → batch pose → hooks → save│
├─────────────────────────────────────────────────────────┤
│  Concrete Drills (drills/*.py)                          │  ← Subclasses
│  on_object_detected / on_pose_estimated /               │
│  compute_drill_metrics / build_overlay / generate_report│
├─────────────────────────────────────────────────────────┤
│  Core Modules (core/*.py)                               │  ← Shared utilities
│  Calibration, Pose Backend, Stable Tracker,             │
│  Keypoint Smoother, Ball Physics, Geometry              │
├─────────────────────────────────────────────────────────┤
│  Visualization (visualization/*.py)                     │  ← Drawing & overlays
│  Skeleton drawing, bounding boxes, metrics panel        │
└─────────────────────────────────────────────────────────┘
```

### Processing Pipeline (per video)

```
Phase 1: Init
  Load video → Load YOLO models (Object + Pose) → Start async writer thread

Phase 2: Batched Loop
  Read N frames → GPU batch object detection → GPU batch pose estimation
  → Per-frame: StableTracker → OneEuro smooth → Calibrate → Drill hooks → Draw → Queue to writer

Phase 3: Export
  Flush writer → generate_report() → Save JSON → FFmpeg compress video
```

---

## 📂 Project Structure

```
ScoutAI/
├── __main__.py              # Direct-run entry point (press ▶️ Run)
├── run_drill.py             # CLI entry point (--drill, --video, --config)
├── run_batch.py             # Batch process all videos in a folder
├── config.py                # DrillConfig dataclass (all hyperparameters)
├── constants.py             # Keypoint indices, skeleton defs, colors
├── base_drill.py            # Template Method base class (DO NOT MODIFY the loop)
│
├── drills/                  # Concrete drill analyzers
│   ├── diamond.py           # Diamond cone agility drill
│   ├── seven_cone.py        # 7-cone dribble drill
│   ├── weakfoot.py          # Weak foot dribble & shoot
│   ├── jump.py              # Vertical / countermovement jump
│   ├── jumping_15.py        # Side-to-side jumps over ball (15s)
│   ├── t_test.py            # T-Test agility drill
│   └── shooting.py          # 3-shot shooting drill through cone gate
│
├── core/                    # Shared building blocks
│   ├── pose_backend.py      # YOLO / MediaPipe pose abstraction
│   ├── calibration.py       # Pixel-to-metre calibration (player height / ball size)
│   ├── keypoint_smoother.py # One Euro Filter for pose jitter reduction
│   ├── stable_tracker.py    # Spatial ID persistence across tracker ID changes
│   ├── ball_physics.py      # Velocity tracking & touch validation
│   ├── geometry.py          # dist2d, box_center, angle_between, IoU
│   ├── homography.py        # Perspective transform from cone positions
│   └── waypoint_tracker.py  # Track player progress through cone journey
│
├── visualization/           # Drawing utilities
│   ├── drawing.py           # Skeleton, bounding boxes, ball-foot line
│   └── overlay.py           # Semi-transparent metrics panel, error banners
│
├── configs/                 # YAML configuration files
│   ├── default.yaml
│   ├── diamond.yaml
│   ├── jump.yaml
│   ├── seven_cone.yaml
│   └── weakfoot.yaml
│
├── weights/                 # Model weights (git-ignored, must be placed manually)
│   ├── best.pt              # Custom-trained YOLO object detection (cone/ball/goal)
│   ├── yolov8l-pose.pt      # YOLOv8-Large pose estimation
│   └── botsort_custom.yaml  # BoT-SORT tracker configuration
│
└── requirements.txt
```

---

## ⚙️ Configuration System

All hyperparameters are centralized in `config.py → DrillConfig`. There are **three ways** to set them, in order of priority:

| Priority | Method | Example |
|---|---|---|
| 1 (highest) | Direct override in `__main__.py` | `DrillConfig(batch_size=32, ...)` |
| 2 | YAML file | `DrillConfig.from_yaml("configs/diamond.yaml")` |
| 3 (lowest) | Dataclass defaults | `batch_size: int = 8` in `config.py` |

### Key Parameters

| Parameter | Default | What it controls |
|---|---|---|
| `batch_size` | 8 | Frames per GPU batch (↑ = faster if VRAM allows) |
| `object_confidence` | 0.5 | YOLO detection threshold |
| `object_imgsz` | 1920 | Object detection resolution (↑ = finds smaller cones) |
| `pose_imgsz` | 1024 | Pose estimation resolution |
| `pose_confidence` | 0.5 | Pose detection threshold |
| `device` | `"cuda"` | `"cuda"` or `"cpu"` |
| `kp_smoothing_factor` | 0.3 | One Euro Filter min_cutoff (↓ = smoother) |
| `drill_params` | `{}` | Dict for drill-specific settings |

---

## 🚀 Running

### 1. Direct Run (recommended for development)
Edit settings at the top of `__main__.py`, then run:
```bash
python -m ScoutAI
```

### 2. CLI Mode
```bash
python run_drill.py --drill diamond --video "path/to/video.mp4" --output ./output
python run_drill.py --list   # show available drills
```

### 3. Batch Processing
Edit settings in `run_batch.py`, then run:
```bash
python run_batch.py
```
Processes all `.mp4` / `.mov` / `.avi` files in the specified input folder.

---

## 🛠️ Installation

```bash
git clone https://github.com/YourUser/ScoutAI.git
cd ScoutAI
pip install -r requirements.txt
```

### Required Model Weights
Place in `weights/` directory:
- `best.pt` — Custom YOLO object detection model (trained on cones, balls, goals)
- `yolov8l-pose.pt` — YOLOv8-Large Pose model ([download from Ultralytics](https://docs.ultralytics.com/models/yolov8/))
- `botsort_custom.yaml` — BoT-SORT tracker config (already included)

### Optional
- **FFmpeg** — for automatic video compression after analysis. Install and add to system PATH, or `pip install imageio-ffmpeg`.
- **CUDA** — for GPU acceleration. Requires NVIDIA GPU + matching CUDA toolkit.

---

## 🔧 How to Add a New Drill

### Step 1: Create the drill file
Create `drills/your_drill.py`:

```python
from ..base_drill import BaseDrillAnalyzer

class YourDrillAnalyzer(BaseDrillAnalyzer):
    drill_name = "your_drill"

    def setup(self):
        """Called once before the loop. Init drill-specific state here."""
        self.my_counter = 0

    def on_object_detected(self, frame_num, class_name, box, stable_id, confidence):
        """Called for EACH detected object (cone, ball, goal) per frame."""
        pass

    def on_pose_estimated(self, frame_num, keypoints, player_box, track_id):
        """Called for EACH detected person per frame. keypoints = (N, 3) array."""
        pass

    def compute_drill_metrics(self, frame_num):
        """Run drill logic AFTER all detections/poses for this frame."""
        pass

    def build_overlay(self, frame_num):
        """Populate self.panel with live metrics for the video overlay."""
        self.panel.clear()
        self.panel.add("My Metric: 42", (0, 255, 0))

    def generate_report(self):
        """Return the final JSON report dict."""
        return {"overall_score": 85, "details": {...}}
```

### Step 2: Register in `run_drill.py`
Add to `DRILL_REGISTRY`:
```python
"your_drill": {
    "module": "scoutai.drills.your_drill",
    "class": "YourDrillAnalyzer",
    "description": "Your drill description",
},
```

### Step 3 (optional): Create a YAML config
Create `configs/your_drill.yaml` with drill-specific parameter overrides.

> **Important:** Never override `run()` in `base_drill.py`. All drill logic goes in the 5 abstract hooks + optional `setup()` and `draw_custom()`.

---

## 🧠 Key Technical Details (for developers)

### Pose Smoothing — One Euro Filter
Located in `core/keypoint_smoother.py`. Replaces the old EMA smoother. Speed-adaptive: heavy smoothing when still (removes jitter), light smoothing when moving (prevents lag). Per-person, per-keypoint, per-axis filters.

### Ball Touch Detection
Drills use a **dual-gate** approach:
1. **Proximity** — ball center within threshold distance of player ankle
2. **Physics** — `BallVelocityTracker` (`core/ball_physics.py`) detects velocity spike OR direction change

Both must be true to register a touch. This eliminates false positives from the ball being near the foot without contact.

### Calibration
`core/calibration.py` converts pixel distances to real-world metres:
- **Primary:** Ball bounding box size vs known football diameter (21.5 cm)
- **Fallback:** Player nose-to-ankle pixel height vs known player height

### Stable ID Tracking
YOLO's BoT-SORT tracker may reassign IDs when objects are temporarily lost. `core/stable_tracker.py` re-associates new tracker IDs with the original stable ID based on spatial proximity.

### Performance Optimizations
- **GPU Batch Inference:** Object detection + pose estimation run on N frames at once
- **FP16 (Half Precision):** ~30-40% faster inference on NVIDIA GPUs
- **Async Video Writer:** Dedicated thread writes frames to disk without blocking inference
- **ROI-only Overlay Blending:** Only copies the small panel region instead of the full frame

### Output Structure
```
output/
└── Technical/
    └── Diamond/
        └── 1_diamond/
            ├── 1_diamond.mp4              # Annotated video
            ├── 1_diamond_compressed.mp4   # FFmpeg compressed version
            └── 1_diamond_report.json      # Full analysis report
```
The directory structure mirrors the input folder structure after the `input/` directory.

---

## 📋 Available Drills

| Drill Key | Analyzer Class | Description |
|---|---|---|
| `diamond` | `DiamondDrillAnalyzer` | 4-cone diamond agility sprint |
| `seven_cone` | `SevenConeDrillAnalyzer` | 7-cone dribble (line of cones + turns) |
| `weakfoot` | `WeakFootAnalyzer` | Weak foot dribble & shoot |
| `jump` | `JumpAnalyzer` | Vertical / countermovement jump test |
| `jumping_15` | `Jumping15Analyzer` | Side-to-side jumps over ball for 15s |
| `t_test` | `TTestAnalyzer` | T-Test agility drill |
| `shooting` | `ShootingDrillAnalyzer` | 3 shots through cone gate at goal |

---

## 📄 License
[Insert your license type here, e.g., MIT]
