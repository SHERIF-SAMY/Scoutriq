# ⚽ ScoutAI — Sports Performance Analysis AI

<div align="center">
  <strong>Advanced Computer Vision & LLM-Powered Sports Analytics Platform</strong><br/>
  <sub>Built with FastAPI · YOLOv8 · MediaPipe · PyTorch (CUDA) · Groq LLM</sub>
</div>

---

## 📖 Project Overview

**ScoutAI** is a production-grade AI system that automates the analysis of football drills and athletic exercises from video footage.

The pipeline works as follows:
1. A video is uploaded via the REST API
2. YOLOv8 detects players, balls, and cones; MediaPipe (or YOLO) extracts body pose keypoints
3. Custom drill modules compute athletic metrics (speed, agility, jump height, accuracy …)
4. FFmpeg re-encodes the annotated output video to H.264 for browser playback
5. Groq LLM (Llama-3) generates personalized coaching feedback from the extracted metrics

---

## 🌟 Supported Drills

| Drill ID | Description |
|---|---|
| `seven_cone` | 7-Cone Dribble — Agility & ball control |
| `diamond` | Diamond Sprint — Acceleration & direction change |
| `weakfoot` | Weak-Foot Dribbling & Shooting |
| `jump` | Vertical / Countermovement Jump Test |
| `jumping_15` | 15-Second Continuous Jumping Test |
| `shooting` | Precision Shooting Test |
| `t_test` | T-Test Agility Drill |

---

## 🏗 Architecture & Folder Structure

```text
ScoutAI_TAM_DEMO_v2/
│
├── api.py                        # 🚀 FastAPI app — all REST endpoints live here
├── Dockerfile                    # 🐳 CUDA 11.8 + cuDNN 8 GPU image
├── docker-compose.yml            # 🐳 Compose config (GPU reservation, volumes, healthcheck)
├── docker-entrypoint.sh          # 🐳 Startup: downloads weights, checks GPU, starts uvicorn
├── requirements_docker.txt       # 📦 Python deps for the Docker image
├── requirements.txt              # 📦 Python deps for local dev
├── .env.example                  # 🔐 Template for environment variables
├── verify_gpu.py                 # 🔍 Quick script to confirm CUDA is accessible
├── output/                       # 📁 Runtime: analyzed videos & JSON artifacts
│
└── Scoutriq_Vision/              # ⚙️ Core Python package
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

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Backend / API | Python 3.10 · FastAPI · Uvicorn |
| Object Detection | Ultralytics YOLOv8 (`best2.pt` custom, `yolov8m-pose.pt`) |
| Pose Estimation | YOLOv8-Pose / MediaPipe |
| GPU Acceleration | PyTorch 2.3 + CUDA 11.8 (cu118 wheels) |
| Video Processing | OpenCV · FFmpeg (libx264 H.264 encoding) |
| LLM / AI | Langchain · ChatGroq · Llama-3.1-8b |
| Config | PyYAML · python-dotenv |
| Containerization | Docker · NVIDIA Container Toolkit |

---

## 🚀 Local Development Setup

### Prerequisites
- Python 3.10+
- FFmpeg in system `PATH`
- NVIDIA GPU (optional — falls back to CPU automatically)
- Groq API Key → [console.groq.com](https://console.groq.com/)

### 1. Clone & Install

```bash
git clone <your-repo-url> scoutai
cd scoutai

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and fill in your values:
```

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Place Model Weights

The pose model (`yolov8m-pose.pt`) is included in the repo.
Place your custom detection model at:

```
Scoutriq_Vision/weights/best2.pt
```

### 4. Start the API

```bash
python -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

- Swagger UI → `http://localhost:8000/docs`
- ReDoc → `http://localhost:8000/redoc`

### 5. CLI Testing (No API Needed)

```bash
# List all registered drills
python Scoutriq_Vision/run_drill.py --list

# Run analysis on a local video
python Scoutriq_Vision/run_drill.py --drill seven_cone --video path/to/video.mp4 --output ./output
```

---

## 🌐 API Reference

### `GET /`
Health check → `{ "message": "ScoutAI API is running" }`

### `GET /drills`
Returns all supported drill IDs and their descriptions.

### `POST /analyze-drill`
Core analysis endpoint.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `video` | File | ✅ | Video file to analyze |
| `drill` | String | ✅ | Drill ID (e.g. `jump`, `t_test`) |
| `player_id` | String | ✅ | Unique player identifier |
| `player_height` | Float | ⬜ | Player height in meters (default `1.75`) |
| `pose_backend` | String | ⬜ | `yolo` or `mediapipe` (default `yolo`) |

**Response:**
```json
{
  "metrics": { "jump_height_cm": 42.3, "flight_time_s": 0.59 },
  "analyzed_video_url": "/output/player_001_jump_analyzed.mp4"
}
```

**Example cURL:**
```bash
curl -X POST http://localhost:8000/analyze-drill \
  -F "video=@sample.mp4" \
  -F "drill=jump" \
  -F "player_id=player_001" \
  -F "pose_backend=yolo" \
  -F "player_height=1.80"
```

---

## 🐳 Docker — Complete Guide

### Prerequisites (any Ubuntu 22.04 server)

```bash
# 1. Install Docker
curl -fsSL https://get.docker.com | bash

# 2. Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 3. Verify GPU is accessible inside Docker
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

---

### Docker Image Layers (Optimized Build Cache)

| Layer | Contents | Rebuild Trigger |
|---|---|---|
| Layer 1 | Ubuntu system packages (Python 3.10, FFmpeg, OpenCV deps) | Almost never |
| Layer 2 | PyTorch 2.3 cu118 wheels (~1.8 GB) | CUDA version change |
| Layer 3 | `requirements_docker.txt` Python deps | Dependency update |
| Layer 4 | Application source code | Every code change |

First build: **~10–15 min**. Subsequent code-only rebuilds: **~30 sec**.

---

### Build & Run

```bash
# Clone the project
git clone <your-repo-url> scoutai && cd scoutai

# Set environment variables
cp .env.example .env
nano .env   # set GROQ_API_KEY

# (Optional) Pre-place custom weights to skip download
mkdir -p Scoutriq_Vision/weights
cp /path/to/best2.pt Scoutriq_Vision/weights/

# Build image
docker compose build

# Start in background
docker compose up -d

# Follow startup logs
docker compose logs -f scoutai
```

**Expected startup output:**
```
================================================
 ScoutAI - Container Startup
================================================
[Weights] yolov8m-pose.pt - already present.
[Weights] yolov8n-pose.pt - OK
[GPU] CUDA available: NVIDIA A100-SXM4-40GB
[GPU] VRAM: 40.0 GB
================================================
 Starting ScoutAI API on port 8000 ...
================================================
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

### Update Deployment (Code Change)

```bash
docker compose down
docker build -t scoutai:gpu .
docker compose up -d
docker compose logs -f scoutai
```

---

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | Groq API key for LLM coaching feedback |
| `CUSTOM_WEIGHTS_URL` | ⬜ Optional | Direct HTTPS URL to download `best2.pt` on startup |
| `NVIDIA_VISIBLE_DEVICES` | ⬜ Optional | `all` (default) or specific GPU index |

---

### Useful Docker Commands

```bash
# Check running container status
docker ps

# Inspect container resource usage
docker stats scoutai_api

# Open a shell inside the running container
docker exec -it scoutai_api bash

# Verify GPU inside container
docker exec -it scoutai_api python3 -c "import torch; print(torch.cuda.get_device_name(0))"

# View container logs
docker compose logs -f scoutai

# Stop & remove container
docker compose down

# Remove image (to force clean rebuild)
docker rmi scoutai:gpu
```

---

## ☁️ RunPod Deployment — Step-by-Step

> البروجكت عندك **Dockerized بالكامل ومجرّب محلياً** — كل اللي محتاجه على RunPod هو: ترفع الملفات وتشغّل `docker compose up`. خلاص.

---

### Step 1 — إنشاء الـ Pod على RunPod

1. ادخل على [runpod.io](https://www.runpod.io/) وسجّل دخول.
2. اختار **"GPU Pod"** → اختار الـ GPU المناسب:

| GPU | VRAM | مناسب لـ |
|---|---|---|
| RTX 3090 / 4090 | 24 GB | تطوير واختبار |
| A40 / A100 40GB | 40 GB | إنتاج |
| RTX A5000 | 24 GB | توازن بين السعر والأداء |

3. في **"Container Image"** اكتب:
   ```
   runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04
   ```
   *(الـ template ده جاهز بـ Docker + NVIDIA Container Toolkit + CUDA 11.8 — مش محتاج تنصّب أي حاجة.)*

4. في **"Expose Ports"** أضف TCP port **`8000`**.

5. في **"Volume Mount"** (Persistent Disk):
   - Container path: `/workspace`
   - Size: **≥ 20 GB** (عشان الـ weights والـ output videos)

6. اضغط **"Deploy"** وانتظر الـ pod يشتغل.

---

### Step 2 — اتصل بالـ Pod عبر SSH

RunPod بيديك الـ SSH command جاهزة في الـ dashboard. شكلها:

```bash
ssh root@<pod-ip> -p <port> -i ~/.ssh/id_rsa
```

أو اضغط **"Connect"** → **"Start Web Terminal"** مباشرة من المتصفح.

---

### Step 3 — ارفع البروجكت من جهازك

> **افتح PowerShell على جهازك** (مش على الـ Pod) ونفّذ الأوامر دي.

#### الطريقة الأولى: `scp` — الأبسط والأضمن ✅

```powershell
# ارفع كل الكود (بيستغرق 1-3 دقايق حسب حجم الملفات)
# استبدل <port> و <pod-ip> بالقيم من الـ RunPod dashboard
scp -r -P <port> `
    "C:\Users\Asus\Desktop\ScoutAI_Work\Tech\ScoutAI_TAM_DEMO_v2" `
    root@<pod-ip>:/workspace/scoutai
```

#### الطريقة الثانية: `rsync` — أسرع وبيكمل لو انقطع الإنترنت ⚡

```powershell
# بيتجاهل .venv و __pycache__ و output تلقائياً
rsync -avz --progress `
  --exclude '.venv' `
  --exclude '__pycache__' `
  --exclude 'output/' `
  --exclude '*.pt' `
  -e "ssh -p <port>" `
  "C:/Users/Asus/Desktop/ScoutAI_Work/Tech/ScoutAI_TAM_DEMO_v2/" `
  root@<pod-ip>:/workspace/scoutai/
```

بعد ما الرفع يخلص، ادخل على الـ Pod وتأكد:

```bash
cd /workspace/scoutai
ls   # المفروض تشوف: api.py  Dockerfile  docker-compose.yml  Scoutriq_Vision/  ...
```

---

### Step 4 — ارفع الـ Model Weights

الـ weights ملفات كبيرة وبترفعها بشكل منفصل من PowerShell على جهازك:

```powershell
# ارفع best2.pt (نموذج الكشف المخصص)
scp -P <port> `
    "C:\Users\Asus\Desktop\ScoutAI_Work\Tech\ScoutAI_TAM_DEMO_v2\Scoutriq_Vision\weights\best2.pt" `
    root@<pod-ip>:/workspace/scoutai/Scoutriq_Vision/weights/

# ارفع yolov8m-pose.pt (نموذج الـ pose)
scp -P <port> `
    "C:\Users\Asus\Desktop\ScoutAI_Work\Tech\ScoutAI_TAM_DEMO_v2\yolov8m-pose.pt" `
    root@<pod-ip>:/workspace/scoutai/Scoutriq_Vision/weights/
```

> **ملاحظة:** لو مش عندك `best2.pt` محلياً، ضيف الـ URL في `.env`:
> ```bash
> CUSTOM_WEIGHTS_URL=https://your-link.com/best2.pt
> ```
> والـ entrypoint هيحمله تلقائياً عند startup.

تحقق على الـ Pod إن الـ weights وصلت:

```bash
ls /workspace/scoutai/Scoutriq_Vision/weights/
# يظهر: best2.pt  yolov8m-pose.pt
```

---

### Step 5 — اعمل `.env` على الـ Pod

```bash
cd /workspace/scoutai

# انسخ من الـ example
cp .env.example .env

# عدّل وضيف الـ Groq key
nano .env
```

```env
GROQ_API_KEY=your_groq_api_key_here
```

---

### Step 6 — شغّل `docker compose up` 🚀

```bash
cd /workspace/scoutai

# بناء الـ image (أول مرة ~10-15 دقيقة)
docker compose build

# شغّل في الـ background
docker compose up -d

# تابع الـ logs
docker compose logs -f scoutai
```

**الـ output المتوقع بعد الـ startup:**

```
================================================
 ScoutAI - Container Startup
================================================
[Weights] yolov8m-pose.pt - already present.
[Weights] best2.pt - already present.
[GPU] CUDA available: NVIDIA A100-SXM4-40GB
[GPU] VRAM: 40.0 GB
================================================
 Starting ScoutAI API on port 8000 ...
================================================
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

### Step 7 — الوصول للـ API

RunPod بيعطيك URL عام للـ port. هتلاقيه في الـ dashboard تحت **"Connect"** → **"HTTP Service"**:

```
https://<pod-id>-8000.proxy.runpod.net/
```

| Endpoint | URL |
|---|---|
| Swagger Docs | `https://<pod-id>-8000.proxy.runpod.net/docs` |
| ReDoc | `https://<pod-id>-8000.proxy.runpod.net/redoc` |
| Health Check | `https://<pod-id>-8000.proxy.runpod.net/` |

**اختبار من جهازك:**

```bash
# Health check
curl https://<pod-id>-8000.proxy.runpod.net/

# تحليل فيديو
curl -X POST https://<pod-id>-8000.proxy.runpod.net/analyze-drill \
  -F "video=@sample.mp4" \
  -F "drill=jump" \
  -F "player_id=player_001" \
  -F "player_height=1.80"
```

---

### تحديث الكود بعد التعديل

لو عدّلت في الكود وعايز تبعت التحديث للـ Pod:

```powershell
# من PowerShell على جهازك — ارفع الكود الجديد
rsync -avz --progress `
  --exclude '.venv' `
  --exclude '__pycache__' `
  --exclude 'output/' `
  --exclude '*.pt' `
  -e "ssh -p <port>" `
  "C:/Users/Asus/Desktop/ScoutAI_Work/Tech/ScoutAI_TAM_DEMO_v2/" `
  root@<pod-ip>:/workspace/scoutai/
```

ثم على الـ Pod:

```bash
cd /workspace/scoutai
docker compose down
docker build -t scoutai:gpu .
docker compose up -d
docker compose logs -f scoutai
```

---

### RunPod Cost Tips 💰

- استخدم **Spot instances** في التطوير — أرخص بـ 70%.
- وقّف الـ Pod لما مش شاغله — بتدفع على وقت التشغيل بس.
- استخدم **Network Volume** للـ `/workspace` عشان الـ weights متتمسحش عند restart.

---



## 🩺 Troubleshooting

| Problem | Solution |
|---|---|
| `CUDA not available` in container | Run `nvidia-smi` on host. Ensure `--gpus all` / `runtime: nvidia` in compose. |
| `libGL.so.1 not found` | Rebuild image — `libgl1` is in the Dockerfile already. |
| `mediapipe` import error | Ensure `numpy<2.0` and `mediapipe==0.10.14` in requirements. |
| `ffmpeg: libx264 not found` | Ubuntu 22.04 apt ships libx264 natively — rebuild from scratch. |
| Weights missing at startup | Set `CUSTOM_WEIGHTS_URL` in `.env` or manually copy to `./Scoutriq_Vision/weights/`. |
| Port already in use | Change `8000:8000` → `8080:8000` in `docker-compose.yml`. |
| `KMP_DUPLICATE_LIB_OK` error | Set env var `KMP_DUPLICATE_LIB_OK=TRUE` (OpenCV/PyTorch thread conflict). |
| Video URL not playable in browser | Ensure FFmpeg `-vcodec libx264` step succeeded; check container logs. |
| LLM returns empty feedback | Verify `GROQ_API_KEY` in `.env`; check Groq free-tier rate limits. |

---

## 👨‍💻 How to Add a New Drill

1. **Create the file** in `Scoutriq_Vision/drills/my_new_drill.py`
2. **Inherit from `BaseDrillAnalyzer`:**
   ```python
   from Scoutriq_Vision.base_drill import BaseDrillAnalyzer

   class MyNewDrillAnalyzer(BaseDrillAnalyzer):
       def __init__(self, config):
           super().__init__(config)
           # Initialize custom state here

       def process_frame(self, frame, frame_number):
           # CV logic — return annotated frame
           return frame

       def get_metrics(self):
           return {"my_metric": 42.0}
   ```
3. **Register it** in `api.py` and `Scoutriq_Vision/run_drill.py` drill routing dictionaries.
4. **(Optional)** Add a `Scoutriq_Vision/configs/my_new_drill.yaml` for drill-specific defaults.
5. **(Optional)** Update the LLM prompt in `Scoutriq_Vision/core/llm_feedback.py`.

---

## 🔑 Key Commands Cheatsheet

```bash
# ── Local Dev ────────────────────────────────────────────
python -m uvicorn api:app --reload --host 127.0.0.1 --port 8000

# ── Docker (full rebuild) ─────────────────────────────────
docker compose down
docker build -t scoutai:gpu .
docker compose up -d
docker compose logs -f scoutai

# ── Docker (code-only update, fast) ──────────────────────
docker compose down && docker build -t scoutai:gpu . && docker compose up -d

# ── GPU Verification ──────────────────────────────────────
python verify_gpu.py
docker exec -it scoutai_api python3 -c "import torch; print(torch.cuda.get_device_name(0))"

# ── Expose via Ngrok (local testing) ─────────────────────
ngrok http 8000
```

---

*Maintained by **Eng. Sherif Samy** — AI Developer @ ScoutAI*