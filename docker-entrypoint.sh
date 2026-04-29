#!/bin/bash
set -e

WEIGHTS_DIR="/app/scoutai/weights"
mkdir -p "$WEIGHTS_DIR" /app/output

echo "================================================"
echo " ScoutAI â€” Container Startup"
echo "================================================"

# â”€â”€ Download yolov8m-pose.pt if not present â”€â”€
if [ ! -f "$WEIGHTS_DIR/yolov8m-pose.pt" ]; then
    echo "[Weights] Downloading yolov8m-pose.pt ..."
    wget -q -O "$WEIGHTS_DIR/yolov8m-pose.pt" \
        "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m-pose.pt" \
    && echo "[Weights] yolov8m-pose.pt â€” OK" \
    || echo "[Weights] WARNING: yolov8m-pose.pt download failed. Mount manually."
else
    echo "[Weights] yolov8m-pose.pt â€” already present."
fi

# â”€â”€ Download yolov8n-pose.pt if not present â”€â”€
if [ ! -f "$WEIGHTS_DIR/yolov8n-pose.pt" ]; then
    echo "[Weights] Downloading yolov8n-pose.pt ..."
    wget -q -O "$WEIGHTS_DIR/yolov8n-pose.pt" \
        "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-pose.pt" \
    && echo "[Weights] yolov8n-pose.pt â€” OK" \
    || echo "[Weights] WARNING: yolov8n-pose.pt download failed."
else
    echo "[Weights] yolov8n-pose.pt â€” already present."
fi

# â”€â”€ Download custom weights (ball/cone model) from env var if set â”€â”€
if [ -n "$CUSTOM_WEIGHTS_URL" ]; then
    if [ ! -f "$WEIGHTS_DIR/best2.pt" ]; then
        echo "[Weights] Downloading best2.pt from CUSTOM_WEIGHTS_URL ..."
        wget -q -O "$WEIGHTS_DIR/best2.pt" "$CUSTOM_WEIGHTS_URL" \
        && echo "[Weights] best2.pt -- OK" \
        || echo "[Weights] WARNING: best2.pt download failed. Check CUSTOM_WEIGHTS_URL."
    else
        echo "[Weights] best2.pt â€” already present."
    fi
fi

# â”€â”€ Print GPU status â”€â”€
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'[GPU] CUDA available: {torch.cuda.get_device_name(0)}')
    print(f'[GPU] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('[GPU] WARNING: CUDA not available â€” running on CPU')
" || echo "[GPU] Could not check GPU status."

echo "================================================"
echo " Starting ScoutAI API on port 8000 ..."
echo "================================================"

# Execute CMD (uvicorn) â€” passes all arguments from docker-compose CMD
exec "$@"
