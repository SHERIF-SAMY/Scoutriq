"""
pose_backend.py — Abstraction layer for pose estimation backends.

Supports both YOLOv8-Pose and MediaPipe BlazePose, allowing drills
to choose the appropriate backend via configuration.

Both backends normalise their output to a common format:
  - keypoints: np.ndarray of shape (N, 3) with [x_pixel, y_pixel, confidence]
  - player_box: (x1, y1, x2, y2)
"""

from __future__ import annotations

import os
import math
from abc import ABC, abstractmethod
from typing import Optional

import cv2
import numpy as np


# ═══════════════════════════════════════════════════════════════
#  Common result dataclass
# ═══════════════════════════════════════════════════════════════

class PoseResult:
    """Normalised pose result from any backend."""

    __slots__ = ("keypoints", "player_box", "confidence", "track_id")

    def __init__(
        self,
        keypoints: np.ndarray,
        player_box: tuple[float, float, float, float],
        confidence: float = 1.0,
        track_id: int = 0,
    ):
        self.keypoints = keypoints      # (N, 3) — [x, y, conf]
        self.player_box = player_box    # (x1, y1, x2, y2)
        self.confidence = confidence
        self.track_id = track_id


# ═══════════════════════════════════════════════════════════════
#  Abstract base
# ═══════════════════════════════════════════════════════════════

class PoseBackend(ABC):
    """Abstract interface for pose estimation."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[PoseResult]:
        """
        Detect poses in a single frame.

        Args:
            frame: BGR image (np.ndarray).

        Returns:
            List of PoseResult, one per detected person.
        """

    @abstractmethod
    def detect_batch(self, frames: list[np.ndarray]) -> list[list[PoseResult]]:
        """
        Detect poses in multiple frames (batch inference).

        Args:
            frames: List of BGR images.

        Returns:
            List of (list of PoseResult), one list per frame.
        """

    @abstractmethod
    def close(self) -> None:
        """Release any held resources."""

    @property
    @abstractmethod
    def num_keypoints(self) -> int:
        """Number of keypoints this backend produces (17 for COCO, 33 for BlazePose)."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Human-readable name: 'yolo' or 'mediapipe'."""


# ═══════════════════════════════════════════════════════════════
#  YOLOv8-Pose backend
# ═══════════════════════════════════════════════════════════════

class YoloPoseBackend(PoseBackend):
    """
    Uses Ultralytics YOLOv8-Pose for 17-keypoint COCO estimation.

    Supports persistent tracking via `.track()` for stable IDs.
    """

    def __init__(
        self,
        model_path: str = "yolov8m-pose.onnx",
        confidence: float = 0.5,
        iou: float = 0.7,
        imgsz: int = 1280,
        device: str = "cpu",
    ):
        from ultralytics import YOLO

        # Explicitly set task='pose' for ONNX
        self.model = YOLO(model_path, task="pose")
        
        # .to() is only for PyTorch .pt models
        if model_path.endswith(".pt"):
            self.model.to(device)
            
        self.device = device # Store for use in track()
        self.confidence = confidence
        self.iou = iou
        self.imgsz = imgsz

    def detect(self, frame: np.ndarray) -> list[PoseResult]:
        return self.detect_batch([frame])[0]

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[PoseResult]]:
        results = self.model.track(
            frames,
            persist=True,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
            device=self.device,
            half=True,  # FP16 inference — ~30-40% faster on NVIDIA GPUs
        )

        all_frame_poses: list[list[PoseResult]] = []
        
        for r in results:
            poses: list[PoseResult] = []
            if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None:
                all_frame_poses.append(poses)
                continue

            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            ids = (
                r.boxes.id.cpu().numpy().astype(int)
                if r.boxes.id is not None
                else list(range(1, len(boxes) + 1))
            )
            all_kps = r.keypoints.data.cpu().numpy()

            for i, (box, conf, tid) in enumerate(zip(boxes, confs, ids)):
                if i < len(all_kps):
                    poses.append(
                        PoseResult(
                            keypoints=all_kps[i],
                            player_box=tuple(box),
                            confidence=float(conf),
                            track_id=int(tid),
                        )
                    )
            all_frame_poses.append(poses)

        return all_frame_poses

    def close(self) -> None:
        pass  # YOLO models don't need explicit cleanup

    @property
    def num_keypoints(self) -> int:
        return 17

    @property
    def backend_name(self) -> str:
        return "yolo"


# ═══════════════════════════════════════════════════════════════
#  MediaPipe BlazePose backend
# ═══════════════════════════════════════════════════════════════

class MediaPipePoseBackend(PoseBackend):
    """
    Uses MediaPipe BlazePose for 33-keypoint estimation with depth (z).

    Supports both the legacy `mp.solutions.pose` API and the newer
    `mediapipe.tasks` API, auto-detecting which is available.

    Note: MediaPipe does not provide bounding boxes, so we compute them
    from the extremes of visible keypoints.
    """

    def __init__(
        self,
        model_complexity: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_path: str = "",
    ):
        import mediapipe as mp

        self._use_legacy = False
        self._pose = None
        self._landmarker = None

        # Try legacy API first
        try:
            _ = mp.solutions.pose
            self._use_legacy = True
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=model_complexity,
                enable_segmentation=False,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        except (AttributeError, ImportError):
            # Fall back to Tasks API
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision as mp_vision

            if not model_path:
                model_path = self._ensure_model()

            base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
            options = mp_vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=min_detection_confidence,
                min_pose_presence_confidence=min_tracking_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
            self._frame_ts = 0
            self._mp = mp

    @staticmethod
    def _ensure_model() -> str:
        """Download the pose_landmarker model if not cached."""
        model_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(model_dir, "..", "pose_landmarker_heavy.task")
        model_path = os.path.normpath(model_path)
        if not os.path.exists(model_path):
            url = (
                "https://storage.googleapis.com/mediapipe-models/"
                "pose_landmarker/pose_landmarker_heavy/float16/latest/"
                "pose_landmarker_heavy.task"
            )
            print(f"   Downloading pose model to {model_path} ...")
            import urllib.request
            urllib.request.urlretrieve(url, model_path)
            print("   Download complete.")
        return model_path

    def detect(self, frame: np.ndarray) -> list[PoseResult]:
        return self.detect_batch([frame])[0]

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[PoseResult]]:
        all_results = []
        for frame in frames:
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            landmarks = None

            if self._use_legacy:
                rgb.flags.writeable = False
                result = self._pose.process(rgb)
                if result.pose_landmarks:
                    landmarks = result.pose_landmarks.landmark
            else:
                import mediapipe as mp
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                self._frame_ts += 33
                result = self._landmarker.detect_for_video(mp_image, self._frame_ts)
                if result.pose_landmarks and len(result.pose_landmarks) > 0:
                    landmarks = result.pose_landmarks[0]

            if landmarks is None:
                all_results.append([])
                continue

            # Convert to (33, 3) array with pixel coordinates
            kps = np.zeros((33, 3), dtype=np.float32)
            for i, lm in enumerate(landmarks):
                kps[i, 0] = lm.x * w
                kps[i, 1] = lm.y * h
                kps[i, 2] = lm.visibility if hasattr(lm, "visibility") else 1.0

            # Compute bounding box from visible keypoints
            visible = kps[kps[:, 2] > 0.3]
            if len(visible) > 0:
                x1 = float(visible[:, 0].min())
                y1 = float(visible[:, 1].min())
                x2 = float(visible[:, 0].max())
                y2 = float(visible[:, 1].max())
                padding = 10
                box = (
                    max(0, x1 - padding),
                    max(0, y1 - padding),
                    min(w, x2 + padding),
                    min(h, y2 + padding),
                )
            else:
                box = (0.0, 0.0, float(w), float(h))

            all_results.append([PoseResult(keypoints=kps, player_box=box, track_id=1)])
        
        return all_results

    def close(self) -> None:
        if self._use_legacy and self._pose:
            self._pose.close()
        elif self._landmarker:
            self._landmarker.close()

    @property
    def num_keypoints(self) -> int:
        return 33

    @property
    def backend_name(self) -> str:
        return "mediapipe"


# ═══════════════════════════════════════════════════════════════
#  Factory
# ═══════════════════════════════════════════════════════════════

def create_pose_backend(
    backend_name: str = "yolo",
    model_path: str = "",
    confidence: float = 0.5,
    imgsz: int = 1280,
    device: str = "cpu",
    **kwargs,
) -> PoseBackend:
    """
    Factory function to create a pose backend by name.

    Args:
        backend_name: 'yolo' or 'mediapipe'
        model_path: Path to model weights.
        confidence: Minimum detection confidence.
        imgsz: Output inference size.

    Returns:
        Configured PoseBackend instance.
    """
    name = backend_name.lower().strip()
    if name == "yolo":
        return YoloPoseBackend(
            model_path=model_path or "yolov8m-pose.onnx",
            confidence=confidence,
            iou=kwargs.get("iou", 0.7),
            imgsz=imgsz,
            device=device,
        )
    elif name in ("mediapipe", "blazepose"):
        return MediaPipePoseBackend(
            model_path=model_path,
            min_detection_confidence=confidence,
            min_tracking_confidence=kwargs.get("tracking_confidence", 0.5),
            model_complexity=kwargs.get("model_complexity", 2),
        )
    else:
        raise ValueError(
            f"Unknown pose backend: '{backend_name}'. Use 'yolo' or 'mediapipe'."
        )
