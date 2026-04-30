"""
base_drill.py — Template Method base class for all drill analyzers.

The main video processing loop (load → detect → pose → draw → save) lives
here and is NOT overridden by subclasses.  Subclasses implement only the
abstract hooks that define drill-specific behaviour:

    on_object_detected()    — handle each detected object (cone, ball, ...)
    on_pose_estimated()     — handle each pose estimation result
    compute_drill_metrics() — run drill-specific logic per frame
    build_overlay()         — populate the metrics panel
    generate_report()       — produce the final JSON report dict
"""

from __future__ import annotations

import os
import json
import cv2
import numpy as np
from abc import ABC, abstractmethod
from queue import Queue, Empty
from datetime import datetime
from threading import Thread
from typing import Optional

from .config import DrillConfig
from .constants import KP_L_ANKLE, KP_R_ANKLE
from .core.calibration import CalibrationManager
from .core.keypoint_smoother import KeypointSmoother
from .core.stable_tracker import StableIDTracker
from .core.pose_backend import create_pose_backend, PoseResult
from .core.geometry import box_center
from .visualization.drawing import draw_pose, draw_custom_box
from .visualization.overlay import MetricsPanel, draw_error_banner


# ═══════════════════════════════════════════════════════════════
#  Module-level model cache — persists across API requests
#  Key: model path (str)  →  Value: loaded model instance
# ═══════════════════════════════════════════════════════════════

_OBJECT_MODEL_CACHE: dict = {}   # model_path → YOLO detect instance
_POSE_BACKEND_CACHE: dict = {}   # "backend:model_path" → PoseBackend instance


class _NumpyEncoder(json.JSONEncoder):
    """Handle numpy types that json.dump can't serialize."""

    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class BaseDrillAnalyzer(ABC):
    """
    Abstract base for all drill analyzers.

    Subclass contract:
        1. Implement the 5 abstract methods below.
        2. Set `drill_name` as a class attribute.
        3. Optionally override `setup()` for drill-specific init.

    The `run()` method handles the full pipeline:
        load video → per-frame detection + pose → hooks → overlay → save
    """

    drill_name: str = "base"

    def __init__(self, config: DrillConfig):
        self.config = config

        # ── Shared components ──
        self.calibration = CalibrationManager(
            player_height_m=config.player_height_m,
            football_diameter_m=config.football_diameter_m,
        )
        self.kp_smoother = KeypointSmoother(
            smoothing_factor=config.kp_smoothing_factor,
            history_size=config.kp_history_size,
        )
        self.object_tracker = StableIDTracker(
            position_threshold=config.stable_tracker_threshold,
        )
        self.player_tracker = StableIDTracker(
            position_threshold=config.player_tracker_threshold,
        )
        self.panel = MetricsPanel()

        # ── State ──
        self.frame_count: int = 0
        self.fps: float = 30.0
        self.frame_width: int = 0
        self.frame_height: int = 0
        self.total_frames: int = 0
        self.current_keypoints: Optional[np.ndarray] = None
        self.current_errors: list[str] = []

        # ── Models (loaded in run()) ──
        self._object_model = None
        self._pose_backend = None

    # ═══════════════════════════════════════════════════════════
    #  ABSTRACT HOOKS — must be implemented by each drill
    # ═══════════════════════════════════════════════════════════

    @abstractmethod
    def on_object_detected(
        self,
        frame_num: int,
        class_name: str,
        box: np.ndarray,
        stable_id: int,
        confidence: float,
    ) -> None:
        """
        Called for each object detection (cone, ball, etc.) per frame.

        Args:
            frame_num: Current frame index (1-based).
            class_name: Detected class name (e.g. 'football', 'cone').
            box: Bounding box (x1, y1, x2, y2).
            stable_id: Spatially-stable tracking ID.
            confidence: Detection confidence [0, 1].
        """

    @abstractmethod
    def on_pose_estimated(
        self,
        frame_num: int,
        keypoints: np.ndarray,
        player_box: np.ndarray,
        track_id: int,
    ) -> None:
        """
        Called for each detected person pose per frame.

        Args:
            frame_num: Current frame index.
            keypoints: Smoothed keypoints (N, 3) — [x, y, conf].
            player_box: (x1, y1, x2, y2) for the player.
            track_id: Stable player tracking ID.
        """

    @abstractmethod
    def compute_drill_metrics(self, frame_num: int) -> None:
        """
        Run drill-specific per-frame logic (touch detection, cone
        contacts, loss-of-control, etc.).

        Called AFTER all detections and poses for this frame.
        """

    @abstractmethod
    def build_overlay(self, frame_num: int) -> None:
        """
        Populate `self.panel` with drill-specific metrics for display.

        Call `self.panel.clear()` first, then `self.panel.add(...)`.
        """

    @abstractmethod
    def generate_report(self) -> dict:
        """
        Generate the final JSON report dictionary.

        Called after all frames have been processed.
        """

    # ═══════════════════════════════════════════════════════════
    #  OPTIONAL HOOKS — override for custom setup / drawing
    # ═══════════════════════════════════════════════════════════

    def setup(self) -> None:
        """
        Called once before the processing loop begins.
        Override to initialise drill-specific state.
        """

    def draw_custom(self, frame: np.ndarray, frame_num: int) -> np.ndarray:
        """
        Called each frame for drill-specific drawing (e.g., cone numbers).
        Override to add custom visuals beyond the standard overlays.
        Default: no-op.
        """
        return frame

    # ═══════════════════════════════════════════════════════════
    #  MAIN PIPELINE  (do not override)
    # ═══════════════════════════════════════════════════════════

    def run(self, video_path: str, output_dir: str) -> dict:
        """
        Execute the full analysis pipeline.

        Args:
            video_path: Path to input video file.
            output_dir: Directory for output video + JSON report.

        Returns:
            The report dictionary (also saved as JSON).
        """
        # ── Validate input ──
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Extract sub-structure after "input" directory to mirror in output
        normalized_video_path = os.path.normpath(video_path)
        sub_structure = ""
        parts = normalized_video_path.split(os.sep)
        if "input" in parts:
            input_idx = parts.index("input")
            sub_dirs = parts[input_idx + 1 : -1]
            if sub_dirs:
                sub_structure = os.path.join(*sub_dirs)

        basename = os.path.splitext(os.path.basename(video_path))[0].strip()
        out_subfolder = os.path.join(output_dir, sub_structure, basename)
        os.makedirs(out_subfolder, exist_ok=True)

        output_video_path = os.path.join(out_subfolder, f"{basename}.mp4")
        output_report_path = os.path.join(out_subfolder, f"{basename}_report.json")

        # ── Open video ──
        cap = cv2.VideoCapture(video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"{'=' * 55}")
        print(f"  ScoutAI — {self.drill_name}")
        print(f"{'=' * 55}")
        print(f"Input : {video_path}")
        print(f"        {self.frame_width}×{self.frame_height} @ {self.fps:.0f}fps, "
              f"{self.total_frames} frames")
        print(f"Output: {output_video_path}\n")

        # ── Load models ──
        print("Loading models...")
        self._load_models()
        print("Models loaded!\n")

        # ── Setup ──
        self.setup()

        # ── Threaded Video Writer ──
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            output_video_path, fourcc, int(self.fps),
            (self.frame_width, self.frame_height),
        )
        write_queue: Queue = Queue()
        writer_running = True

        def _writer_thread():
            """Background thread that writes frames to disk."""
            while writer_running or not write_queue.empty():
                try:
                    frame_to_write = write_queue.get(timeout=0.1)
                    out.write(frame_to_write)
                except Empty:
                    pass

        writer = Thread(target=_writer_thread, daemon=True)
        writer.start()

        class_names = self._object_model.names if self._object_model else {}
        batch_size = max(1, self.config.batch_size)

        # ── Frame loop (batched) ──
        print(f"Processing video (batch size {batch_size})...\n")
        self.frame_count = 0

        while cap.isOpened():
            # ── Read a batch of frames ──
            raw_frames = []
            for _ in range(batch_size):
                ret, frame = cap.read()
                if not ret:
                    break
                raw_frames.append(frame)

            if not raw_frames:
                break

            actual_batch_size = len(raw_frames)

            # ── Pad the batch to keep ONNX input shape constant ──
            # If the last batch is smaller than batch_size, ONNX Runtime will re-allocate memory
            # and re-benchmark CUDA kernels, causing a 10+ second delay. Padding fixes this.
            if actual_batch_size < batch_size:
                pad_frames = [np.zeros_like(raw_frames[0])] * (batch_size - actual_batch_size)
                inference_frames = raw_frames + pad_frames
            else:
                inference_frames = raw_frames

            # ── Batch GPU inference ──
            obj_results = self._get_batch_objects(inference_frames, class_names)[:actual_batch_size]
            pose_results = self._get_batch_poses(inference_frames)[:actual_batch_size]

            # ── Per-frame sequential post-processing ──
            for i, frame in enumerate(raw_frames):
                self.frame_count += 1
                self.current_keypoints = None
                self.current_errors = []

                # 1. Object post-processing
                frame = self._post_process_objects(
                    frame, obj_results[i], class_names,
                )

                # 2. Pose post-processing
                frame = self._post_process_pose(frame, pose_results[i])

                # 2b. Live calibration
                self.calibration.compute()

                # 3. Drill-specific metrics
                self.compute_drill_metrics(self.frame_count)

                # 4. Custom drawing
                frame = self.draw_custom(frame, self.frame_count)

                # 5. Metrics panel
                self.build_overlay(self.frame_count)
                frame = self.panel.draw(frame)

                # 6. Error banners
                frame = draw_error_banner(frame, self.current_errors)

                # 7. Queue frame for async writing
                write_queue.put(frame)

                if self.frame_count % 10 == 0:
                    pct = 100 * self.frame_count / max(self.total_frames, 1)
                    print(f"  Processed {self.frame_count}/{self.total_frames} ({pct:.1f}%)")

        # Wait for writer to finish
        writer_running = False
        writer.join()

        cap.release()
        out.release()

        # ── Finalize calibration ──
        self.calibration.compute()

        # ── Generate report ──
        print(f"\n{'=' * 55}")
        print(f"  GENERATING REPORT")
        print(f"{'=' * 55}")

        report = self.generate_report()

        # Add common metadata
        report.setdefault("drill_info", {})
        report["drill_info"].update({
            "drill_type": self.drill_name,
            "video": os.path.basename(video_path),
            "timestamp": datetime.now().isoformat(),
            "total_frames": self.total_frames,
            "fps": int(self.fps),
            "resolution": f"{self.frame_width}x{self.frame_height}",
            "calibration": {
                "is_calibrated": self.calibration.is_calibrated,
                "pixels_per_meter": (
                    round(self.calibration.pixels_per_meter, 2)
                    if self.calibration.is_calibrated
                    else None
                ),
                "source": self.calibration.source,
            },
        })

        # ── Save report ──
        with open(output_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)

        # ── Compress Video ──
        compressed_video_path = os.path.join(out_subfolder, f"{basename}_compressed.mp4")
        print(f"\nCompressing video to save space... (may take a moment)")
        try:
            import subprocess
            import shutil
            
            ffmpeg_path = shutil.which("ffmpeg")
            if not ffmpeg_path:
                try:
                    import imageio_ffmpeg
                    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
                except ImportError:
                    pass
                    
            if not ffmpeg_path:
                print("⚠️ FFmpeg not found in PATH. Skipping compression. Please install FFmpeg and add it to your System PATH.")
                print(f"✅ Original Video : {output_video_path}")
            else:
                # libx264 with crf 28 (high compression, good quality), preset fast
                subprocess.run([
                    ffmpeg_path, "-y", "-i", output_video_path,
                    "-vcodec", "libx264", "-crf", "28", "-preset", "fast",
                    compressed_video_path
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"✅ Original Video : {output_video_path}")
                print(f"✅ Compressed Video: {compressed_video_path}")
        except Exception as e:
            print(f"⚠️ Failed to compress video: {e}")
            print(f"✅ Original Video : {output_video_path}")

        print(f"📄 Report saved to: {output_report_path}")

        # ── Cleanup ──
        # Only close pose backend if it is NOT a shared cached instance.
        # Cached backends must stay alive for reuse across requests.
        if self._pose_backend and self._pose_backend not in _POSE_BACKEND_CACHE.values():
            self._pose_backend.close()

        return report

    # ═══════════════════════════════════════════════════════════
    #  INTERNAL HELPERS
    # ═══════════════════════════════════════════════════════════

    def _load_models(self) -> None:
        """Load YOLO object detection and pose estimation models.
        
        Models are cached at the module level so they are only loaded once
        per server lifetime — subsequent requests reuse the cached instances.
        """
        from ultralytics import YOLO

        if self.config.object_model_path:
            obj_key = self.config.object_model_path
            if obj_key in _OBJECT_MODEL_CACHE:
                print(f"[Model Cache HIT]  Reusing cached object model: {obj_key}")
                self._object_model = _OBJECT_MODEL_CACHE[obj_key]
            else:
                print(f"[Model Cache MISS] Loading object model: {obj_key}")
                model = YOLO(obj_key, task="detect")
                # .to() is only for .pt files. For ONNX, device is passed in predict/track.
                if obj_key.endswith(".pt"):
                    model.to(self.config.device)
                _OBJECT_MODEL_CACHE[obj_key] = model
                self._object_model = model
                print(f"[Model Cache]      Object model cached ✓")

        pose_key = f"{self.config.pose_backend}:{self.config.pose_model_path}"
        if pose_key in _POSE_BACKEND_CACHE:
            print(f"[Model Cache HIT]  Reusing cached pose backend: {pose_key}")
            self._pose_backend = _POSE_BACKEND_CACHE[pose_key]
        else:
            print(f"[Model Cache MISS] Loading pose backend: {pose_key}")
            backend = create_pose_backend(
                backend_name=self.config.pose_backend,
                model_path=self.config.pose_model_path,
                confidence=self.config.pose_confidence,
                iou=self.config.pose_iou,
                imgsz=self.config.pose_imgsz,
                device=self.config.device,
            )
            _POSE_BACKEND_CACHE[pose_key] = backend
            self._pose_backend = backend
            print(f"[Model Cache]      Pose backend cached ✓")

    def _get_batch_objects(self, frames: list[np.ndarray], class_names: dict) -> list:
        """Run object detection on a batch of frames."""
        if self._object_model is None:
            return [None] * len(frames)

        tracker_arg = (
            self.config.tracker_config_path
            if self.config.tracker_config_path and os.path.exists(self.config.tracker_config_path)
            else None
        )

        kwargs = dict(
            persist=True,
            conf=self.config.object_confidence,
            verbose=False,
            imgsz=self.config.object_imgsz,
            device=self.config.device,
            half=True,  # FP16 inference — ~30-40% faster on NVIDIA GPUs
        )
        if tracker_arg:
            kwargs["tracker"] = tracker_arg

        return self._object_model.track(frames, **kwargs)

    def _post_process_objects(self, frame: np.ndarray, result, class_names: dict) -> np.ndarray:
        """Handle individual frame results from batched object detection."""
        if result is None or result.boxes is None or len(result.boxes) == 0:
            self.object_tracker.end_frame()
            return frame

        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = (
            result.boxes.id.cpu().numpy()
            if result.boxes.id is not None
            else list(range(len(boxes)))
        )
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()

        for box, tid, cid, conf in zip(boxes, track_ids, class_ids, confidences):
            cname = class_names.get(cid, f"cls{cid}")

            if cname.lower() in getattr(self.config, "goal_class_names", []) and not self.config.drill_params.get("track_goals", False):
                continue

            if cname.lower() in self.config.ball_class_names:
                stable_id = 1
                self.calibration.update_from_ball_box(box)
            else:
                stable_id = self.object_tracker.update(int(tid), box, cid)

            self.on_object_detected(self.frame_count, cname, box, stable_id, float(conf))

            label = f"ID:{stable_id} {cname} {conf:.0%}"
            frame = draw_custom_box(frame, box, label, class_id=cid)

        self.object_tracker.end_frame()
        return frame

    def _get_batch_poses(self, frames: list[np.ndarray]) -> list[list[PoseResult]]:
        """Run pose estimation on a batch of frames."""
        if self._pose_backend is None:
            return [[]] * len(frames)
        return self._pose_backend.detect_batch(frames)

    def _post_process_pose(self, frame: np.ndarray, poses: list[PoseResult]) -> np.ndarray:
        """Handle individual frame results from batched pose estimation."""
        for pose in poses:
            stable_pid = self.player_tracker.update(pose.track_id, pose.player_box, -1)
            smoothed = self.kp_smoother.smooth(stable_pid, pose.keypoints)
            self.current_keypoints = smoothed

            self.calibration.update_from_keypoints(smoothed, self.config.keypoint_confidence)
            self.on_pose_estimated(self.frame_count, smoothed, pose.player_box, stable_pid)

            skeleton_type = "coco" if self._pose_backend.backend_name == "yolo" else "mediapipe"
            frame = draw_pose(frame, smoothed, skeleton_type=skeleton_type)

        self.player_tracker.end_frame()
        return frame

    def _process_objects(self, frame: np.ndarray, class_names: dict) -> np.ndarray:
        """Legacy single-frame wrapper."""
        result = self._get_batch_objects([frame], class_names)[0]
        return self._post_process_objects(frame, result, class_names)

    def _process_pose(self, frame: np.ndarray) -> np.ndarray:
        """Legacy single-frame wrapper."""
        poses = self._get_batch_poses([frame])[0]
        return self._post_process_pose(frame, poses)
