import argparse
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import gltflib
import mediapipe as mp
import numpy as np
import pyaudio
import torch
try:
    import pyrender.pyrender as pyrender
except Exception:
    import pyrender  # fallback for environments where top-level package is correct

from abs_test import (
    ARKIT_BLENDSHAPE_NAMES,
    BEAT_TO_METADATA_IDX,
    AudioToBlendshapeModel,
)

np.infty = np.inf
os.environ["PYOPENGL_ERROR_CHECKING"] = "0"

# Ordered outer-lip loop:
# left corner -> upper lip -> right corner -> lower lip -> back toward left
MOUTH_TRACK_INDICES = [
    61,
    185,
    40,
    39,
    37,
    0,
    267,
    269,
    270,
    409,
    291,
    375,
    321,
    405,
    314,
    17,
    84,
    181,
    91,
    146,
]


@dataclass
class SourcePatch:
    image: np.ndarray
    alpha: np.ndarray
    points: np.ndarray


class GLBFaceRenderer:
    """Offscreen GLB renderer with morph target weight updates."""

    def __init__(self, glb_path: str, width: int, height: int, log_callback=print):
        self.log = log_callback
        self.w = int(width)
        self.h = int(height)
        self.metadata_name_to_idx = {name: i for i, name in enumerate(ARKIT_BLENDSHAPE_NAMES)}

        glb_path = Path(glb_path).resolve()
        if not glb_path.exists():
            raise FileNotFoundError(f"Avatar model not found: {glb_path}")

        self.log(f"[GLBRenderer] Loading avatar: {glb_path}")
        gltf = gltflib.GLTF.load(str(glb_path))
        self.scene = pyrender.Scene.from_gltflib_scene(gltf)
        self.mesh_target_names = self._build_mesh_target_name_map(gltf)

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=self.w / self.h)
        cam_pose = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 1.3],
                [0.0, 0.0, 1.0, 0.4],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        self.scene.add(camera, pose=cam_pose)
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        self.scene.add(light, pose=cam_pose)

        self.morph_nodes = []
        for node in self.scene.nodes:
            if node.mesh is None:
                continue
            for prim in node.mesh.primitives:
                has_targets = hasattr(prim, "targets") and prim.targets is not None
                if not has_targets or not hasattr(prim.targets, "positions") or prim.targets.positions is None:
                    continue

                num_blendshapes = int(prim.targets.positions.shape[0])
                node.mesh.weights = np.zeros(num_blendshapes, dtype=np.float32)
                mesh_name = node.name or node.mesh.name or "Unnamed_Mesh"
                target_names = self.mesh_target_names.get(mesh_name)
                if target_names and len(target_names) != num_blendshapes:
                    target_names = None
                self.morph_nodes.append((node, num_blendshapes, target_names))
                break

        if not self.morph_nodes:
            self.log("[GLBRenderer] Warning: No morph targets found in avatar.")

        self.renderer = pyrender.OffscreenRenderer(self.w, self.h)

    def _build_mesh_target_name_map(self, gltf):
        mesh_target_names = {}
        model = gltf.model
        if model is None or not model.meshes:
            return mesh_target_names

        for mesh_def in model.meshes:
            mesh_name = getattr(mesh_def, "name", None) or "Unnamed_Mesh"
            target_names = self._extract_target_names(mesh_def)
            if target_names:
                mesh_target_names[mesh_name] = target_names
        return mesh_target_names

    @staticmethod
    def _extract_target_names(mesh_def):
        def names_from_extras(extras):
            if not isinstance(extras, dict):
                return None
            names = extras.get("targetNames")
            if isinstance(names, list) and names:
                return [str(n) for n in names]
            return None

        names = names_from_extras(getattr(mesh_def, "extras", None))
        if names:
            return names

        primitives = getattr(mesh_def, "primitives", None) or []
        for prim in primitives:
            names = names_from_extras(getattr(prim, "extras", None))
            if names:
                return names
        return None

    def render(self, blendshape_weights):
        model_weights = np.asarray(blendshape_weights, dtype=np.float32)
        metadata_weights = np.zeros(len(ARKIT_BLENDSHAPE_NAMES), dtype=np.float32)
        for beat_idx, metadata_idx in BEAT_TO_METADATA_IDX.items():
            if beat_idx < len(model_weights):
                metadata_weights[metadata_idx] = float(model_weights[beat_idx])

        for node, expected_length, target_names in self.morph_nodes:
            if target_names:
                weights = np.zeros(expected_length, dtype=np.float32)
                for target_idx, target_name in enumerate(target_names):
                    if target_idx >= expected_length:
                        break
                    metadata_idx = self.metadata_name_to_idx.get(target_name)
                    if metadata_idx is not None and metadata_idx < len(metadata_weights):
                        weights[target_idx] = float(metadata_weights[metadata_idx])
            else:
                weights = np.zeros(expected_length, dtype=np.float32)
                copy_len = min(expected_length, len(metadata_weights))
                if copy_len > 0:
                    weights[:copy_len] = metadata_weights[:copy_len]
            node.mesh.weights = weights

        color, _ = self.renderer.render(self.scene, flags=pyrender.RenderFlags.RGBA)
        return cv2.cvtColor(color, cv2.COLOR_RGBA2BGR)

    def close(self):
        if self.renderer is not None:
            try:
                self.renderer.delete()
            except Exception:
                pass
            self.renderer = None


class ABSStreamSource:
    """Audio-driven blendshape source + GLB avatar frame renderer."""

    def __init__(
        self,
        model_path: str,
        avatar_model_path: str,
        frame_width: int = 640,
        frame_height: int = 720,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        window_seconds: float = 0.4,
        smoothing: float = 0.35,
        face_landmarker_task: str = ".cache/face_landmarker.task",
        log_callback=print,
    ):
        self.model_path = Path(model_path)
        self.avatar_model_path = str(avatar_model_path)
        self.w = int(frame_width)
        self.h = int(frame_height)
        self.log = log_callback

        self.RATE = int(sample_rate)
        self.CHUNK = int(chunk_size)
        self.CHANNELS = 1
        self.FORMAT = pyaudio.paInt16

        self.window_chunks = max(3, int((self.RATE * float(window_seconds)) / self.CHUNK))
        self.smoothing = float(np.clip(smoothing, 0.0, 0.95))

        self.audio_queue = queue.Queue(maxsize=3)
        self.is_running = False
        self.p = pyaudio.PyAudio()
        self.stream = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._configure_torch_backends()
        self.model = None
        self.last_blendshape = np.zeros(52, dtype=np.float32)
        self.latest_blendshape = np.zeros(52, dtype=np.float32)
        self.bs_name_to_idx = {name: i for i, name in enumerate(ARKIT_BLENDSHAPE_NAMES)}
        self._threads = []
        self._state_lock = threading.Lock()
        self._source_timestamp_ms = 0
        self._source_smoothed_points = None
        self._source_miss_count = 0
        self._source_max_miss_frames = 12
        self._source_backend = None
        self._source_face_mesh = None
        self._source_landmarker = None

        self._load_model()
        self.avatar_renderer = GLBFaceRenderer(self.avatar_model_path, self.w, self.h, self.log)
        self._init_source_mouth_tracker(face_landmarker_task)

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self._threads = [
            threading.Thread(target=self._stream_audio, daemon=True),
            threading.Thread(target=self._inference_loop, daemon=True),
        ]
        for t in self._threads:
            t.start()
        self.log("[ABSSource] Started audio stream + inference threads.")

    def stop(self):
        self.is_running = False
        if self.stream is not None:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

        try:
            self.p.terminate()
        except Exception:
            pass

        for t in self._threads:
            if t.is_alive():
                t.join(timeout=0.5)
        self._threads = []

        if self._source_face_mesh is not None:
            try:
                self._source_face_mesh.close()
            except Exception:
                pass
            self._source_face_mesh = None

        if self._source_landmarker is not None:
            try:
                self._source_landmarker.close()
            except Exception:
                pass
            self._source_landmarker = None

        if hasattr(self, "avatar_renderer") and self.avatar_renderer is not None:
            self.avatar_renderer.close()
        self.log("[ABSSource] Stopped.")

    def get_latest(self):
        """Returns: (abs_frame_bgr, blendshape_vec, source_mouth_points_2d)."""
        with self._state_lock:
            vals = self.latest_blendshape.copy()
        frame, src_points = self._render_abs_frame(vals)
        return frame, vals, src_points

    def _configure_torch_backends(self):
        if self.device.type != "cpu":
            return
        nnpack_backend = getattr(torch.backends, "nnpack", None)
        if nnpack_backend is None:
            return
        try:
            nnpack_backend.enabled = False
        except Exception:
            pass

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device)
        config = checkpoint.get("config", {})
        hidden_dim = config.get("hidden_dim", 512)
        use_pretrained = config.get("use_pretrained", False)

        self.model = AudioToBlendshapeModel(
            hidden_dim=hidden_dim,
            use_pretrained=use_pretrained,
        ).to(self.device)

        state_dict = checkpoint.get("model_state_dict")
        if state_dict is None:
            raise KeyError("Checkpoint missing 'model_state_dict'.")

        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.log(
            f"[ABSSource] Loaded model: {self.model_path.name} "
            f"(hidden_dim={hidden_dim}, use_pretrained={use_pretrained}, device={self.device})"
        )

    def _stream_audio(self):
        try:
            self.stream = self.p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK,
            )

            rolling = deque(maxlen=self.window_chunks)
            while self.is_running:
                data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                rolling.append(data)
                if len(rolling) < self.window_chunks:
                    continue
                try:
                    if self.audio_queue.full():
                        _ = self.audio_queue.get_nowait()
                    self.audio_queue.put_nowait(list(rolling))
                except queue.Empty:
                    pass
                except queue.Full:
                    pass
        except Exception as e:
            self.log(f"[ABSSource] Audio stream error: {e}")
        finally:
            if self.stream is not None:
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                except Exception:
                    pass
                self.stream = None

    def _frames_to_audio(self, frames, gain=15.0):
        audio_int16 = np.frombuffer(b"".join(frames), dtype=np.int16)
        if len(audio_int16) == 0:
            return None
        audio = (audio_int16.astype(np.float32) / 32768.0) * float(gain)
        audio = np.clip(audio, -1.0, 1.0)
        return torch.from_numpy(audio).unsqueeze(0)

    def _predict_blendshape(self, audio_tensor):
        with torch.no_grad():
            out = self.model(audio_tensor.to(self.device))
            blendshape = out[0, -1].detach().cpu().numpy().astype(np.float32)
        self.last_blendshape = self.smoothing * self.last_blendshape + (1.0 - self.smoothing) * blendshape
        return self.last_blendshape.copy()

    def _inference_loop(self):
        while self.is_running:
            try:
                frames = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                audio = self._frames_to_audio(frames)
                if audio is None:
                    continue
                blendshape = self._predict_blendshape(audio)
                with self._state_lock:
                    self.latest_blendshape = blendshape
            except Exception as e:
                self.log(f"[ABSSource] Inference error: {e}")

    def _init_source_mouth_tracker(self, face_landmarker_task: str):
        # Prefer legacy solutions API when available.
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            self._source_backend = "solutions"
            self._source_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.log("[ABSSource] Using mediapipe.solutions for source mouth tracking.")
            return

        self._source_backend = "tasks"
        task_path = Path(face_landmarker_task)
        if not task_path.is_absolute():
            project_root = Path(__file__).resolve().parents[1]
            task_path = project_root / task_path
        task_path = task_path.resolve()
        if not task_path.exists():
            raise FileNotFoundError(
                f"Face landmarker task file not found: {task_path}. "
                "Pass a valid path via --face-landmarker-task."
            )

        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(task_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._source_landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self.log(f"[ABSSource] Using mediapipe.tasks for source mouth tracking ({task_path}).")

    def _detect_source_mouth_points(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        points = None

        if self._source_backend == "solutions":
            results = self._source_face_mesh.process(rgb)
            if not results.multi_face_landmarks:
                self._source_miss_count += 1
                if self._source_smoothed_points is not None:
                    return self._source_smoothed_points.copy()
                return None
            lm = results.multi_face_landmarks[0].landmark
            h, w = frame_bgr.shape[:2]
            points = np.asarray([[lm[idx].x * w, lm[idx].y * h] for idx in MOUTH_TRACK_INDICES], dtype=np.float32)
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            self._source_timestamp_ms += 1
            result = self._source_landmarker.detect_for_video(mp_image, self._source_timestamp_ms)
            if not result.face_landmarks:
                self._source_miss_count += 1
                if self._source_smoothed_points is not None:
                    return self._source_smoothed_points.copy()
                return None
            lm = result.face_landmarks[0]
            h, w = frame_bgr.shape[:2]
            points = np.asarray([[lm[idx].x * w, lm[idx].y * h] for idx in MOUTH_TRACK_INDICES], dtype=np.float32)

        self._source_miss_count = 0
        if self._source_smoothed_points is None:
            self._source_smoothed_points = points
        else:
            self._source_smoothed_points = 0.55 * self._source_smoothed_points + 0.45 * points
        return self._source_smoothed_points.copy()

    def _fallback_source_mouth_points(self):
        cx = self.w * 0.50
        cy = self.h * 0.62
        rx = self.w * 0.115
        ry = self.h * 0.065
        t_upper = np.linspace(np.pi, 0.0, 11, endpoint=True)
        t_lower = np.linspace(0.0, np.pi, 11, endpoint=True)[1:-1]
        t = np.concatenate([t_upper, t_lower]).astype(np.float32)
        x = cx + rx * np.cos(t)
        y = cy + ry * np.sin(t)
        return np.stack([x, y], axis=1).astype(np.float32)

    def _mouth_controls(self, vals):
        def bs(name):
            idx = self.bs_name_to_idx.get(name)
            if idx is None or idx >= len(vals):
                return 0.0
            return float(np.clip(vals[idx], 0.0, 1.0))

        jaw_open = bs("jawOpen")
        smile_left = bs("mouthSmileLeft")
        smile_right = bs("mouthSmileRight")
        smile = 0.5 * (smile_left + smile_right)
        pucker = bs("mouthPucker")
        funnel = bs("mouthFunnel")
        wide = 0.5 * (bs("mouthStretchLeft") + bs("mouthStretchRight"))
        upper = 0.5 * (bs("mouthUpperUpLeft") + bs("mouthUpperUpRight"))
        lower = 0.5 * (bs("mouthLowerDownLeft") + bs("mouthLowerDownRight"))
        close = bs("mouthClose")
        return {
            "jaw": jaw_open,
            "smile": smile,
            "smile_left": smile_left,
            "smile_right": smile_right,
            "pucker": 0.65 * pucker + 0.35 * funnel,
            "wide": wide,
            "upper": upper,
            "lower": lower,
            "close": close,
        }

    def _mouth_xyz(self, ctrl, t):
        width = 1.2 + 0.9 * ctrl["wide"] + 0.35 * ctrl["smile"] - 0.45 * ctrl["pucker"]
        height = 0.20 + 1.2 * ctrl["jaw"] + 0.45 * ctrl["lower"] - 0.35 * ctrl["close"]
        depth = 0.1 + 0.9 * ctrl["pucker"]

        x = width * np.cos(t)
        y = 0.9 * height * np.sin(t)
        z = 0.33 * depth * np.cos(2 * t)

        upper_mask = np.sin(t) > 0
        y[upper_mask] -= 0.15 * ctrl["upper"]
        y[~upper_mask] += 0.18 * ctrl["lower"]
        y += 0.23 * ctrl["smile"] * np.sign(np.cos(t)) * np.abs(np.cos(t))
        return np.stack([x, y, z], axis=1)

    def _build_mouth_rings_3d(self, ctrl, n=40):
        t = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=False)
        outer = self._mouth_xyz(ctrl, t)
        inner_scale = 0.45 + 0.3 * ctrl["jaw"]
        inner = outer * np.array([inner_scale, inner_scale * 0.85, 0.75], dtype=np.float32)
        inner[:, 2] -= 0.2 + 0.2 * ctrl["pucker"]
        return outer, inner

    def _build_ordered_source_points_3d(self, ctrl):
        t_upper = np.linspace(np.pi, 0.0, 11, endpoint=True)
        t_lower = np.linspace(0.0, np.pi, 11, endpoint=True)[1:-1]
        t = np.concatenate([t_upper, t_lower]).astype(np.float32)
        return self._mouth_xyz(ctrl, t)

    def _project_points(self, pts, yaw=0.0, pitch=0.0):
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
        rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float32)

        pts_r = (pts @ ry.T) @ rx.T
        z = pts_r[:, 2] + 4.8
        f = min(self.w, self.h) * 0.43
        px = self.w * 0.5 + f * (pts_r[:, 0] / z)
        py = self.h * 0.54 + f * (pts_r[:, 1] / z)
        return np.stack([px, py], axis=1).astype(np.float32), pts_r

    def _render_abs_frame(self, vals):
        frame = self.avatar_renderer.render(vals)
        source_points = self._detect_source_mouth_points(frame)
        if source_points is None:
            source_points = self._fallback_source_mouth_points()
        return frame, source_points


class WebcamFaceTracker:
    """Webcam reader + MediaPipe mouth landmark tracker with EMA smoothing."""

    def __init__(
        self,
        webcam_index: int = 0,
        width: int = 640,
        height: int = 720,
        landmark_smoothing: float = 0.6,
        face_landmarker_task: str = ".cache/face_landmarker.task",
        log_callback=print,
    ):
        self.log = log_callback
        self.webcam_index = int(webcam_index)
        self.width = int(width)
        self.height = int(height)
        self.landmark_smoothing = float(np.clip(landmark_smoothing, 0.0, 0.95))
        self._timestamp_ms = 0
        self._miss_count = 0
        self._max_miss_frames = 12

        self.cap = cv2.VideoCapture(self.webcam_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open webcam index {self.webcam_index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.backend = None
        self.face_mesh = None
        self.landmarker = None

        # Prefer legacy solutions API when available.
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            self.backend = "solutions"
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.log("[FaceTracker] Using mediapipe.solutions.face_mesh backend.")
        else:
            # Fallback to Tasks API (some mediapipe builds expose only mp.tasks).
            self.backend = "tasks"
            task_path = Path(face_landmarker_task)
            if not task_path.is_absolute():
                project_root = Path(__file__).resolve().parents[1]
                task_path = project_root / task_path
            task_path = task_path.resolve()
            if not task_path.exists():
                raise FileNotFoundError(
                    f"Face landmarker task file not found: {task_path}. "
                    "Pass a valid path via --face-landmarker-task."
                )

            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            options = mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(task_path)),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)
            self.log(f"[FaceTracker] Using mediapipe.tasks FaceLandmarker backend ({task_path}).")

        self._smoothed_mouth_points = None

    def read_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def track_mouth(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        points = None

        if self.backend == "solutions":
            results = self.face_mesh.process(rgb)
            if not results.multi_face_landmarks:
                self._miss_count += 1
                if self._smoothed_mouth_points is not None and self._miss_count <= self._max_miss_frames:
                    return self._smoothed_mouth_points.copy()
                self._smoothed_mouth_points = None
                return None
            lm = results.multi_face_landmarks[0].landmark
            h, w = frame_bgr.shape[:2]
            points = np.asarray([[lm[idx].x * w, lm[idx].y * h] for idx in MOUTH_TRACK_INDICES], dtype=np.float32)
        else:
            # Tasks API path.
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            self._timestamp_ms += 1
            result = self.landmarker.detect_for_video(mp_image, self._timestamp_ms)
            if not result.face_landmarks:
                self._miss_count += 1
                if self._smoothed_mouth_points is not None and self._miss_count <= self._max_miss_frames:
                    return self._smoothed_mouth_points.copy()
                self._smoothed_mouth_points = None
                return None
            lm = result.face_landmarks[0]
            h, w = frame_bgr.shape[:2]
            points = np.asarray([[lm[idx].x * w, lm[idx].y * h] for idx in MOUTH_TRACK_INDICES], dtype=np.float32)

        self._miss_count = 0
        if self._smoothed_mouth_points is None:
            self._smoothed_mouth_points = points
        else:
            a = self.landmark_smoothing
            self._smoothed_mouth_points = a * self._smoothed_mouth_points + (1.0 - a) * points

        return self._smoothed_mouth_points.copy()

    def stop(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        if self.face_mesh is not None:
            try:
                self.face_mesh.close()
            except Exception:
                pass
            self.face_mesh = None

        if self.landmarker is not None:
            try:
                self.landmarker.close()
            except Exception:
                pass
            self.landmarker = None


class MouthOverlayMapper:
    """Piecewise-affine mouth warper + natural alpha blending."""

    def __init__(
        self,
        overlay_alpha: float = 0.95,
        feather_px: int = 9,
        overlay_scale: float = 1.12,
        style_strength: float = 0.0,
        min_alpha: float = 0.35,
        min_mouth_height_px: float = 6.0,
        draw_outline: bool = True,
    ):
        self.overlay_alpha = float(np.clip(overlay_alpha, 0.0, 1.0))
        self.feather_px = int(max(1, feather_px))
        self.overlay_scale = float(np.clip(overlay_scale, 0.9, 1.5))
        self.style_strength = float(np.clip(style_strength, 0.0, 1.0))
        self.min_alpha = float(np.clip(min_alpha, 0.0, 1.0))
        self.min_mouth_height_px = float(max(1.0, min_mouth_height_px))
        self.draw_outline = bool(draw_outline)
        self._last_valid_src_points = None
        self._last_valid_dst_points = None
        self._last_good_patch = None

    def build_source_patch(self, abs_frame, source_mouth_points):
        if abs_frame is None or source_mouth_points is None:
            return self._last_good_patch

        pts = np.asarray(source_mouth_points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
            return self._last_good_patch
        pts = self._ensure_min_mouth_height(pts, self.min_mouth_height_px)

        x, y, w, h = cv2.boundingRect(np.round(pts).astype(np.int32))
        pad = 10
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(abs_frame.shape[1], x + w + pad)
        y1 = min(abs_frame.shape[0], y + h + pad)
        if x1 <= x0 or y1 <= y0:
            return self._last_good_patch

        patch = abs_frame[y0:y1, x0:x1].copy()
        local_pts = pts - np.array([x0, y0], dtype=np.float32)

        # Make the transferred mouth more visually obvious than subtle lip tinting.
        patch = self._style_source_patch(patch)

        mask = np.zeros((patch.shape[0], patch.shape[1]), dtype=np.float32)
        local_poly = np.round(local_pts).astype(np.int32)
        contour_area = abs(float(cv2.contourArea(local_poly.astype(np.float32))))
        if contour_area < 3.0:
            hull = cv2.convexHull(local_poly)
            cv2.fillConvexPoly(mask, hull, 1.0, lineType=cv2.LINE_AA)
            cv2.polylines(mask, [local_poly], True, 1.0, 3, cv2.LINE_AA)
        else:
            cv2.fillPoly(mask, [local_poly], 1.0, lineType=cv2.LINE_AA)

        k = self.feather_px * 2 + 1
        mask = cv2.GaussianBlur(mask, (3, 3), 0.5)
        alpha = cv2.GaussianBlur(mask, (k, k), self.feather_px * 0.6)
        alpha = np.clip(alpha * 1.35, 0.0, 1.0)
        alpha = np.maximum(alpha, mask * self.min_alpha)
        candidate = SourcePatch(image=patch, alpha=alpha, points=local_pts)
        if self._is_good_patch(candidate):
            self._last_good_patch = candidate
            return candidate
        return self._last_good_patch if self._last_good_patch is not None else candidate

    def overlay(self, frame, patch_data, dst_points):
        if patch_data is None or dst_points is None:
            return frame

        src_pts = np.asarray(patch_data.points, dtype=np.float32)
        dst_pts = np.asarray(dst_points, dtype=np.float32)
        if src_pts.shape != dst_pts.shape or src_pts.shape[0] < 3:
            return frame
        src_pts = self._ensure_min_mouth_height(src_pts, self.min_mouth_height_px)
        dst_pts = self._ensure_min_mouth_height(dst_pts, self.min_mouth_height_px)
        src_pts = self._stabilize_polygon(src_pts, is_source=True)
        dst_pts = self._stabilize_polygon(dst_pts, is_source=False)
        if src_pts is None or dst_pts is None:
            return frame

        if abs(self.overlay_scale - 1.0) > 1e-4:
            center = np.mean(dst_pts, axis=0, keepdims=True)
            dst_pts = (dst_pts - center) * self.overlay_scale + center

        tri_indices = self._ear_clip_triangulation(src_pts)
        if not tri_indices:
            tri_indices = self._fan_triangulation(src_pts.shape[0])
        h, w = frame.shape[:2]
        accum_img = np.zeros((h, w, 3), dtype=np.float32)
        accum_alpha = np.zeros((h, w), dtype=np.float32)

        for tri in tri_indices:
            self._warp_triangle(
                patch_data.image,
                patch_data.alpha,
                src_pts[list(tri)],
                dst_pts[list(tri)],
                accum_img,
                accum_alpha,
            )

        if not np.any(accum_alpha > 0.0):
            return frame

        composed = np.zeros_like(accum_img, dtype=np.float32)
        valid = accum_alpha > 1e-5
        composed[valid] = accum_img[valid] / accum_alpha[valid, None]
        alpha = np.clip(accum_alpha, 0.0, 1.0) * self.overlay_alpha

        out = frame.astype(np.float32)
        out = out * (1.0 - alpha[..., None]) + composed * alpha[..., None]
        out = np.clip(out, 0.0, 255.0).astype(np.uint8)

        if self.draw_outline:
            cv2.polylines(
                out,
                [np.round(dst_pts).astype(np.int32)],
                True,
                (255, 190, 70),
                2,
                cv2.LINE_AA,
            )
        return out

    def _style_source_patch(self, patch):
        if self.style_strength <= 1e-6:
            return patch
        patch_f = patch.astype(np.float32)
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        # Bright pink-ish tint so the mapping effect is unmistakable on webcam.
        tint = np.zeros_like(patch_f)
        tint[..., 0] = 195.0
        tint[..., 1] = 70.0
        tint[..., 2] = 245.0

        styled = patch_f * (1.0 - self.style_strength) + tint * self.style_strength
        styled *= (0.75 + 0.55 * gray[..., None])
        return np.clip(styled, 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def _is_good_patch(patch_data):
        if patch_data is None:
            return False
        alpha = np.asarray(patch_data.alpha, dtype=np.float32)
        image = np.asarray(patch_data.image)
        if alpha.size == 0 or image.size == 0:
            return False
        mask = alpha > 0.45
        if int(np.count_nonzero(mask)) < 40:
            return False
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        region = gray[mask]
        if region.size < 40:
            return False
        # Reject near-flat patches which usually come from a tracking miss on avatar source.
        return float(np.std(region)) >= 2.0

    @staticmethod
    def _ensure_min_mouth_height(points, min_height_px):
        pts = np.asarray(points, dtype=np.float32).copy()
        if pts.ndim != 2 or pts.shape[0] < 3:
            return pts
        y_min = float(np.min(pts[:, 1]))
        y_max = float(np.max(pts[:, 1]))
        height = y_max - y_min
        if height >= float(min_height_px):
            return pts
        center_y = float(np.mean(pts[:, 1]))
        scale = float(min_height_px) / max(height, 1e-3)
        pts[:, 1] = (pts[:, 1] - center_y) * scale + center_y
        return pts

    def _stabilize_polygon(self, points, is_source):
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
            return None
        if not np.all(np.isfinite(pts)):
            pts = None
        else:
            area = abs(float(self._signed_area(pts)))
            if area < 5.0 or (not self._is_simple_polygon(pts)):
                pts = None

        if is_source:
            if pts is not None:
                self._last_valid_src_points = pts.copy()
                return pts
            if self._last_valid_src_points is not None:
                return self._last_valid_src_points.copy()
            return None

        if pts is not None:
            self._last_valid_dst_points = pts.copy()
            return pts
        if self._last_valid_dst_points is not None:
            return self._last_valid_dst_points.copy()
        return None

    @staticmethod
    def _fan_triangulation(n_points):
        return [(0, i, i + 1) for i in range(1, n_points - 1)]

    @classmethod
    def _ear_clip_triangulation(cls, pts):
        pts = np.asarray(pts, dtype=np.float32)
        n = int(pts.shape[0])
        if n < 3:
            return []
        if n == 3:
            return [(0, 1, 2)]

        is_ccw = cls._signed_area(pts) > 0.0
        idx = list(range(n))
        triangles = []
        guard = 0
        max_iter = n * n

        while len(idx) > 3 and guard < max_iter:
            guard += 1
            ear_found = False
            m = len(idx)
            for i in range(m):
                i_prev = idx[(i - 1) % m]
                i_curr = idx[i]
                i_next = idx[(i + 1) % m]
                a, b, c = pts[i_prev], pts[i_curr], pts[i_next]
                if not cls._is_convex(a, b, c, is_ccw):
                    continue
                if abs(float(cls._tri_area2(a, b, c))) < 1e-4:
                    continue
                contains_other = False
                for j in idx:
                    if j in (i_prev, i_curr, i_next):
                        continue
                    if cls._point_in_triangle(pts[j], a, b, c):
                        contains_other = True
                        break
                if contains_other:
                    continue
                triangles.append((i_prev, i_curr, i_next))
                del idx[i]
                ear_found = True
                break
            if not ear_found:
                return []

        if len(idx) == 3:
            triangles.append((idx[0], idx[1], idx[2]))
        return triangles

    @staticmethod
    def _signed_area(pts):
        x = pts[:, 0]
        y = pts[:, 1]
        return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

    @staticmethod
    def _tri_area2(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    @classmethod
    def _is_convex(cls, a, b, c, is_ccw):
        cross = float(cls._tri_area2(a, b, c))
        if is_ccw:
            return cross > 1e-5
        return cross < -1e-5

    @classmethod
    def _point_in_triangle(cls, p, a, b, c):
        s1 = cls._tri_area2(p, a, b)
        s2 = cls._tri_area2(p, b, c)
        s3 = cls._tri_area2(p, c, a)
        has_neg = (s1 < -1e-6) or (s2 < -1e-6) or (s3 < -1e-6)
        has_pos = (s1 > 1e-6) or (s2 > 1e-6) or (s3 > 1e-6)
        return not (has_neg and has_pos)

    @classmethod
    def _is_simple_polygon(cls, pts):
        n = int(pts.shape[0])
        if n < 3:
            return False

        def seg_intersect(a, b, c, d):
            def orient(p, q, r):
                return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

            def on_seg(p, q, r):
                return (
                    min(p[0], r[0]) - 1e-6 <= q[0] <= max(p[0], r[0]) + 1e-6
                    and min(p[1], r[1]) - 1e-6 <= q[1] <= max(p[1], r[1]) + 1e-6
                )

            o1 = orient(a, b, c)
            o2 = orient(a, b, d)
            o3 = orient(c, d, a)
            o4 = orient(c, d, b)

            if (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0):
                return True
            if abs(o1) <= 1e-6 and on_seg(a, c, b):
                return True
            if abs(o2) <= 1e-6 and on_seg(a, d, b):
                return True
            if abs(o3) <= 1e-6 and on_seg(c, a, d):
                return True
            if abs(o4) <= 1e-6 and on_seg(c, b, d):
                return True
            return False

        for i in range(n):
            a1 = pts[i]
            a2 = pts[(i + 1) % n]
            for j in range(i + 1, n):
                if j == i or (j + 1) % n == i or j == (i + 1) % n:
                    continue
                # Skip the first/last shared vertex edge pair.
                if i == 0 and j == n - 1:
                    continue
                b1 = pts[j]
                b2 = pts[(j + 1) % n]
                if seg_intersect(a1, a2, b1, b2):
                    return False
        return True

    @staticmethod
    def _warp_triangle(src_img, src_alpha, src_tri, dst_tri, accum_img, accum_alpha):
        src_tri = np.asarray(src_tri, dtype=np.float32)
        dst_tri = np.asarray(dst_tri, dtype=np.float32)

        r1 = cv2.boundingRect(src_tri)
        r2 = cv2.boundingRect(dst_tri)
        if r1[2] <= 0 or r1[3] <= 0 or r2[2] <= 0 or r2[3] <= 0:
            return

        x1, y1, w1, h1 = r1
        x2, y2, w2, h2 = r2
        src_rect = src_tri - np.array([x1, y1], dtype=np.float32)
        dst_rect = dst_tri - np.array([x2, y2], dtype=np.float32)
        if abs(float(cv2.contourArea(src_rect))) < 0.02 or abs(float(cv2.contourArea(dst_rect))) < 0.02:
            return

        src_roi = src_img[y1 : y1 + h1, x1 : x1 + w1]
        alpha_roi = src_alpha[y1 : y1 + h1, x1 : x1 + w1]
        if src_roi.size == 0 or alpha_roi.size == 0:
            return

        warp_mat = cv2.getAffineTransform(src_rect, dst_rect)
        warped_img = cv2.warpAffine(
            src_roi,
            warp_mat,
            (w2, h2),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(np.float32)
        warped_alpha = cv2.warpAffine(
            alpha_roi,
            warp_mat,
            (w2, h2),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(np.float32)

        tri_mask = np.zeros((h2, w2), dtype=np.float32)
        cv2.fillConvexPoly(tri_mask, np.round(dst_rect).astype(np.int32), 1.0, lineType=cv2.LINE_AA)
        warped_alpha *= tri_mask

        h_acc, w_acc = accum_alpha.shape[:2]
        if x2 >= w_acc or y2 >= h_acc:
            return
        x2c = max(0, x2)
        y2c = max(0, y2)
        x2e = min(w_acc, x2 + w2)
        y2e = min(h_acc, y2 + h2)
        if x2e <= x2c or y2e <= y2c:
            return

        ox0 = x2c - x2
        oy0 = y2c - y2
        ox1 = ox0 + (x2e - x2c)
        oy1 = oy0 + (y2e - y2c)

        alpha_part = warped_alpha[oy0:oy1, ox0:ox1]
        img_part = warped_img[oy0:oy1, ox0:ox1]

        accum_img[y2c:y2e, x2c:x2e] += img_part * alpha_part[..., None]
        accum_alpha[y2c:y2e, x2c:x2e] += alpha_part


class VideoMapApp:
    """Orchestrates ABS stream (left) + webcam mouth overlay mapping (right)."""

    def __init__(self, args):
        self.args = args
        self.abs_source = ABSStreamSource(
            model_path=args.model,
            avatar_model_path=args.avatar,
            frame_width=args.width,
            frame_height=args.height,
            window_seconds=args.window_seconds,
            smoothing=args.smoothing,
            face_landmarker_task=args.face_landmarker_task,
        )
        self.face_tracker = WebcamFaceTracker(
            webcam_index=args.webcam_index,
            width=args.width,
            height=args.height,
            landmark_smoothing=args.landmark_smoothing,
            face_landmarker_task=args.face_landmarker_task,
        )
        self.mapper = MouthOverlayMapper(
            overlay_alpha=args.overlay_alpha,
            overlay_scale=args.overlay_scale,
            style_strength=args.overlay_style_strength,
            draw_outline=not args.no_overlay_outline,
        )

    def run(self):
        self.abs_source.start()
        frame_interval = 1.0 / max(1, int(self.args.fps))
        window_name = "video_map_test (Left: ABS | Right: Webcam+Mouth Overlay)"

        try:
            while True:
                loop_start = time.time()
                webcam_frame = self.face_tracker.read_frame()
                if webcam_frame is None:
                    raise RuntimeError("Failed to read webcam frame.")

                abs_frame, _, src_mouth_points = self.abs_source.get_latest()
                dst_mouth_points = self.face_tracker.track_mouth(webcam_frame)

                patch_data = self.mapper.build_source_patch(abs_frame, src_mouth_points)
                if patch_data is not None and dst_mouth_points is not None:
                    mapped_frame = self.mapper.overlay(webcam_frame, patch_data, dst_mouth_points)
                else:
                    mapped_frame = webcam_frame

                left = cv2.resize(abs_frame, (self.args.width, self.args.height), interpolation=cv2.INTER_LINEAR)
                right = cv2.resize(mapped_frame, (self.args.width, self.args.height), interpolation=cv2.INTER_LINEAR)

                cv2.putText(left, "ABS Stream", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (235, 240, 255), 2, cv2.LINE_AA)
                cv2.putText(
                    right,
                    "Webcam + ABS Mouth Map",
                    (14, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (235, 240, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(right, "Press q to quit", (14, self.args.height - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 230, 245), 1, cv2.LINE_AA)

                canvas = np.hstack([left, right])
                cv2.imshow(window_name, canvas)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                elapsed = time.time() - loop_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            self.face_tracker.stop()
            self.abs_source.stop()
            cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="ABS mouth -> webcam mouth mapping test.")
    parser.add_argument("--model", required=True, default="models/best_fast.pt", help="Path to ABS checkpoint .pt")
    parser.add_argument("--avatar", required=True, default="assets/avatar.glb", help="Path to GLB avatar model used as mouth source")
    parser.add_argument("--webcam-index", type=int, default=0, help="OpenCV webcam index")
    parser.add_argument("--width", type=int, default=640, help="Per-panel output width")
    parser.add_argument("--height", type=int, default=720, help="Per-panel output height")
    parser.add_argument("--fps", type=int, default=30, help="Target loop FPS")
    parser.add_argument("--window-seconds", type=float, default=0.4, help="Audio window length fed to ABS model")
    parser.add_argument("--smoothing", type=float, default=0.35, help="ABS blendshape EMA smoothing factor")
    parser.add_argument("--overlay-alpha", type=float, default=0.95, help="Mouth overlay blend strength (0..1)")
    parser.add_argument("--overlay-scale", type=float, default=1.12, help="Scale mouth overlay region around tracked center")
    parser.add_argument("--overlay-style-strength", type=float, default=0.0, help="How strongly to stylize/tint the transferred mouth (0..1)")
    parser.add_argument("--no-overlay-outline", action="store_true", help="Disable destination lip outline overlay")
    parser.add_argument("--landmark-smoothing", type=float, default=0.6, help="Face landmark EMA smoothing factor")
    parser.add_argument(
        "--face-landmarker-task",
        type=str,
        default=".cache/face_landmarker.task",
        help="Path to MediaPipe FaceLandmarker .task file (used when mp.solutions is unavailable).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    app = VideoMapApp(args)
    app.run()


if __name__ == "__main__":
    main()
