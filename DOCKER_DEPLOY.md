# ScoutAI — Docker GPU Deployment Guide

## Prerequisites

### On any Ubuntu 22.04 GPU server (RunPod / Azure / bare metal):

```bash
# 1. Install Docker
curl -fsSL https://get.docker.com | bash

# 2. Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 3. Verify GPU is accessible
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

---

## Build & Run (Local / Server)

```bash
# Clone / copy project files to server
git clone <your-repo-url> scoutai && cd scoutai

# Set environment variables
cp .env.example .env
# Edit .env: add GROQ_API_KEY and any other required vars
nano .env

# (Optional) Place your custom weights in:
mkdir -p scoutai/weights
cp /path/to/best.pt scoutai/weights/
cp /path/to/best2.pt scoutai/weights/

# Build image (first time ~10–15 min, subsequent builds ~30s)
docker compose build

# Start service
docker compose up -d

# Follow logs
docker compose logs -f scoutai
```

**Expected startup output:**
```
[Weights] yolov8m-pose.pt — already present.
[GPU] CUDA available: NVIDIA A100-SXM4-40GB
[GPU] VRAM: 40.0 GB
Starting ScoutAI API on port 8000 ...
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## Test Endpoints

```bash
# Health check
curl http://localhost:8000/

# List drills
curl http://localhost:8000/drills

# Analyze a video
curl -X POST http://localhost:8000/analyze-drill \
  -F "video=@sample.mp4" \
  -F "drill=jump" \
  -F "player_id=player_001" \
  -F "pose_backend=yolo" \
  -F "player_height=1.80"
```

---

## Environment Variables (`.env`)

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | Groq API key for LLM feedback |
| `CUSTOM_WEIGHTS_URL` | ⬜ Optional | Direct URL to download `best.pt` at startup |

---

## RunPod Deployment

1. Create a pod with CUDA 11.8+ template (e.g., `runpod/pytorch:2.1.0-py3.10-cuda11.8.0`)
2. Set TCP port `8000` exposed
3. SSH into pod and follow **Build & Run** steps above

---

## Azure GPU VM Deployment

```bash
# Create NC-series VM (NC6s_v3 = 1x V100 16GB)
az vm create \
  --resource-group scoutai-rg \
  --name scoutai-gpu \
  --image Canonical:0001-com-ubuntu-server-jammy:22_04-lts:latest \
  --size Standard_NC6s_v3 \
  --admin-username azureuser \
  --generate-ssh-keys

# Install NVIDIA driver + Container Toolkit on the VM
# Then follow Build & Run steps
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `CUDA not available` | Run `nvidia-smi` inside container. Check `--gpus all` flag. |
| `libGL.so.1 not found` | Install `libgl1` via apt (already in Dockerfile) |
| `mediapipe import error` | Ensure `numpy<2.0` and `mediapipe==0.10.14` |
| `ffmpeg: libx264 not found` | Rebuild image — Ubuntu 22.04 apt includes libx264 |
| Weights missing | Set `CUSTOM_WEIGHTS_URL` env var or mount manually to `./scoutai/weights/` |
| Port already in use | Change host port in `docker-compose.yml` from `8000:8000` to e.g. `8080:8000` |


# Run Docker:
docker compose down; docker build -t scoutai:gpu .; docker compose up -d
http://localhost:8000/

# Update Docker:
docker compose down
docker build -t scoutai:gpu .
docker compose up -d
docker compose logs -f scoutai
