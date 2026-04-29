"""
verify_gpu.py — Confirm that ScoutAI GPU acceleration is active.

Usage:
    python verify_gpu.py               # runs inference on a synthetic frame
    python verify_gpu.py test.jpg      # runs inference on a real image
"""
import sys
import numpy as np

print("=" * 55)
print("  ScoutAI GPU Verification Script")
print("=" * 55)

# 1. Check torch + CUDA
import torch
print(f"PyTorch version  : {torch.__version__}")
print(f"CUDA available   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU              : {torch.cuda.get_device_name(0)}")
    print(f"CUDA version     : {torch.version.cuda}")
    print(f"cuDNN version    : {torch.backends.cudnn.version()}")
else:
    print("No CUDA GPU detected — will run on CPU.")

print()

# 2. Check gpu_utils helper
from scoutai.core.gpu_utils import get_device, get_device_label, configure_torch
device = get_device()
configure_torch(device)
print(f"Active device    : {device}")
print(f"Status           : {get_device_label()}")
print()

# 3. Load YOLO and run prediction
from ultralytics import YOLO
model = YOLO("yolov8m-pose.pt")
if device == "cuda":
    model = model.to(device)
    try:
        model.model.half()
        print("FP16 (half precision) : enabled")
    except Exception as e:
        print(f"FP16 not available   : {e}")
else:
    print("FP16 (half precision) : disabled (CPU mode)")

image_path = sys.argv[1] if len(sys.argv) > 1 else None
if image_path is None:
    # Synthetic 640×480 BGR frame
    import cv2
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    print("\nRunning inference on synthetic frame...")
    results = model.predict(frame, device=device, verbose=False)
else:
    print(f"\nRunning inference on: {image_path}")
    results = model.predict(image_path, device=device, verbose=False)

print("Inference complete — no errors.")
print()
print("=" * 55)
print("  ALL CHECKS PASSED" if device == "cuda" else "  Running in CPU mode (CUDA not available)")
print("=" * 55)
