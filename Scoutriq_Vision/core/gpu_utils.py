"""
gpu_utils.py — GPU detection and torch configuration helpers.
Used by api.py at startup to configure the device and print a banner.
"""
import torch

def get_device() -> str:
    """Return 'cuda' if a CUDA GPU is available, otherwise 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"

def configure_torch(device: str) -> None:
    """Apply recommended torch settings for the given device."""
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

def get_device_label() -> str:
    """Return a human-readable GPU/CPU banner string."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
        return f"[GPU] {name} — {vram} GB VRAM"
    return "[CPU] CUDA not available — running on CPU"
