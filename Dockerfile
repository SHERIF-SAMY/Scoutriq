# ════════════════════════════════════════════════════════════════
#  Base: CUDA 11.8 + cuDNN 8 runtime (Ubuntu 22.04)
#  Compatible with PyTorch 2.x cu118 wheels
#  Using 'runtime' not 'devel' → saves ~2.5GB
# ════════════════════════════════════════════════════════════════
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# ── Prevent interactive prompts during apt ──
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ─────────────────────────────────────────────────────────────────
# LAYER 1: System packages (changes rarely → cached longest)
# Installs: Python 3.10, FFmpeg (with libx264), OpenCV deps,
#           MediaPipe deps, curl for healthchecks
# ─────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-distutils \
    # FFmpeg — Ubuntu 22.04 repo ships libx264 built-in
    ffmpeg \
    # OpenCV headless runtime deps
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    # MediaPipe runtime deps
    libgoogle-glog0v5 \
    libprotobuf23 \
    # General utilities
    curl \
    wget \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && python3 -m pip install --upgrade pip setuptools wheel --no-cache-dir

# ─────────────────────────────────────────────────────────────────
# LAYER 2: PyTorch CUDA wheel (largest layer ~1.8GB)
# Installed separately to cache independently of other requirements.
# Rebuild only if CUDA version changes — rare.
# ─────────────────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    torch==2.3.0+cu118 \
    torchvision==0.18.0+cu118 \
    torchaudio==2.3.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# ─────────────────────────────────────────────────────────────────
# LAYER 3: Python dependencies (requirements_docker.txt)
# Cached until requirements file changes.
# ─────────────────────────────────────────────────────────────────
WORKDIR /app
COPY requirements_docker.txt .
RUN pip install --no-cache-dir -r requirements_docker.txt

# ─────────────────────────────────────────────────────────────────
# LAYER 4: Application source code (changes frequently)
# Comes last — rebuilds only when source code changes.
# ─────────────────────────────────────────────────────────────────
COPY . .

# Create required runtime directories
RUN mkdir -p /app/output /app/Scoutriq_Vision/weights

# ─────────────────────────────────────────────────────────────────
# Generate the entrypoint script INSIDE the container using printf.
# This avoids all Windows CRLF/BOM encoding issues from host files.
# ─────────────────────────────────────────────────────────────────
RUN printf '#!/bin/bash\nset -e\n\n\
    WEIGHTS_DIR="/app/Scoutriq_Vision/weights"\n\
    mkdir -p "$WEIGHTS_DIR" /app/output\n\n\
    echo "================================================"\n\
    echo " ScoutAI - Container Startup"\n\
    echo "================================================"\n\n\
    if [ ! -f "$WEIGHTS_DIR/yolov8m-pose.pt" ]; then\n\
    echo "[Weights] Downloading yolov8m-pose.pt ..."\n\
    wget -q -O "$WEIGHTS_DIR/yolov8m-pose.pt" \\\n\
    "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m-pose.pt" \\\n\
    && echo "[Weights] yolov8m-pose.pt - OK" \\\n\
    || echo "[Weights] WARNING: yolov8m-pose.pt download failed. Mount manually."\n\
    else\n\
    echo "[Weights] yolov8m-pose.pt - already present."\n\
    fi\n\n\
    if [ ! -f "$WEIGHTS_DIR/yolov8n-pose.pt" ]; then\n\
    echo "[Weights] Downloading yolov8n-pose.pt ..."\n\
    wget -q -O "$WEIGHTS_DIR/yolov8n-pose.pt" \\\n\
    "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-pose.pt" \\\n\
    && echo "[Weights] yolov8n-pose.pt - OK" \\\n\
    || echo "[Weights] WARNING: yolov8n-pose.pt download failed."\n\
    else\n\
    echo "[Weights] yolov8n-pose.pt - already present."\n\
    fi\n\n\
    if [ -n "$CUSTOM_WEIGHTS_URL" ] && [ ! -f "$WEIGHTS_DIR/best.pt" ]; then\n\
    echo "[Weights] Downloading best.pt from CUSTOM_WEIGHTS_URL ..."\n\
    wget -q -O "$WEIGHTS_DIR/best.pt" "$CUSTOM_WEIGHTS_URL" \\\n\
    && echo "[Weights] best.pt - OK" \\\n\
    || echo "[Weights] WARNING: best.pt download failed."\n\
    fi\n\n\
    python3 -c "\nimport torch\nif torch.cuda.is_available():\n    print('"'"'[GPU] CUDA available:'"'"', torch.cuda.get_device_name(0))\n    print('"'"'[GPU] VRAM:'"'"', round(torch.cuda.get_device_properties(0).total_memory/1024**3,1), '"'"'GB'"'"')\nelse:\n    print('"'"'[GPU] WARNING: CUDA not available - running on CPU'"'"')\n" || echo "[GPU] Could not check GPU status."\n\n\
    echo "================================================"\n\
    echo " Starting ScoutAI API on port 8000 ..."\n\
    echo "================================================"\n\n\
    exec "$@"\n' > /usr/local/bin/docker-entrypoint.sh \
    && chmod +x /usr/local/bin/docker-entrypoint.sh

# ─────────────────────────────────────────────────────────────────
# Expose API port
# ─────────────────────────────────────────────────────────────────
EXPOSE 8000

# ─────────────────────────────────────────────────────────────────
# Entrypoint: downloads weights if missing, then starts uvicorn
# ─────────────────────────────────────────────────────────────────
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
