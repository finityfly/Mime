import argparse
import queue
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pyaudio
import torch
import pyrender
import os
import gltflib

from utils.inference_engine_fast import InferenceEngineFast

def choose_audio_input_device(logger) -> int | None:
    pa = pyaudio.PyAudio()
    devices = []
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if int(info.get("maxInputChannels", 0)) > 0:
            devices.append((i, info.get("name", f"Device {i}")))

    logger("[ABS] Available audio input devices:")
    for idx, name in devices:
        logger(f"  {idx}: {name}")

    try:
        import sys as _sys
        if not _sys.stdin.isatty():
            logger("[ABS] stdin is not interactive; using default input device.")
            return None
    except Exception:
        pass

    try:
        choice = input("Select audio input device index (blank for default): ").strip()
    except Exception:
        return None

    if choice == "":
        return None
    try:
        return int(choice)
    except Exception:
        logger("[ABS] Invalid selection; using default input device.")
        return None

np.product = np.prod 
os.environ["PYOPENGL_ERROR_CHECKING"] = "0"  

ARKIT_BLENDSHAPE_NAMES = [
    'eyeBlinkLeft', 'eyeLookDownLeft', 'eyeLookInLeft', 'eyeLookOutLeft', 'eyeLookUpLeft', 
    'eyeSquintLeft', 'eyeWideLeft', 'eyeBlinkRight', 'eyeLookDownRight', 'eyeLookInRight', 
    'eyeLookOutRight', 'eyeLookUpRight', 'eyeSquintRight', 'eyeWideRight', 'jawForward', 
    'jawLeft', 'jawRight', 'jawOpen', 'mouthClose', 'mouthFunnel', 'mouthPucker', 
    'mouthRight', 'mouthLeft', 'mouthSmileLeft', 'mouthSmileRight', 'mouthFrownRight', 
    'mouthFrownLeft', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthStretchLeft', 
    'mouthStretchRight', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 
    'mouthShrugUpper', 'mouthPressLeft', 'mouthPressRight', 'mouthLowerDownLeft', 
    'mouthLowerDownRight', 'mouthUpperUpLeft', 'mouthUpperUpRight', 'browDownLeft', 
    'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight', 'cheekPuff', 
    'cheekSquintLeft', 'cheekSquintRight', 'noseSneerLeft', 'noseSneerRight', 'tongueOut'
]

MODEL_OUTPUT_BLENDSHAPE_NAMES = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
    "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight", "mouthClose",
    "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel",
    "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight",
    "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower",
    "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight", "tongueOut"
]

_model_name_to_idx = {name: i for i, name in enumerate(MODEL_OUTPUT_BLENDSHAPE_NAMES)}
_metadata_name_to_idx = {name: i for i, name in enumerate(ARKIT_BLENDSHAPE_NAMES)}
BEAT_TO_METADATA_IDX = {
    beat_idx: _metadata_name_to_idx[name]
    for name, beat_idx in _model_name_to_idx.items()
    if name in _metadata_name_to_idx
}


class AudioToBlendshapeModel(torch.nn.Module):
    def __init__(self, num_blendshapes=52, hidden_dim=512, use_pretrained=False):
        super().__init__()
        self.use_pretrained = use_pretrained

        self.audio_backbone = torch.nn.Sequential(
            torch.nn.Conv1d(1, 64, kernel_size=10, stride=4, padding=3),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Conv1d(64, 128, kernel_size=4, stride=4, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(128, 256, kernel_size=4, stride=4, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(256, hidden_dim, kernel_size=4, stride=4, padding=1),
            torch.nn.ReLU(),
        )

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            batch_first=True,
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(256, num_blendshapes),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        if self.use_pretrained:
            features = self.audio_backbone(x).last_hidden_state
            x = self.audio_proj(features)
        else:
            if x.dim() == 2:
                x = x.unsqueeze(1)
            x = self.audio_backbone(x).transpose(1, 2)

        x = self.transformer(x)
        return self.regressor(x)

class GLBEngine:
    def __init__(self, glb_path, w=580, h=720):
        self.w, self.h = w, h
        self.log = print
        self.metadata_name_to_idx = {name: i for i, name in enumerate(ARKIT_BLENDSHAPE_NAMES)}
        
        self.log(f"[GLB] Loading {glb_path}...")
        gltf = gltflib.GLTF.load(str(glb_path))
        self.scene = pyrender.Scene.from_gltflib_scene(gltf)
        self.mesh_target_names = self._build_mesh_target_name_map(gltf)
        
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=w/h)
        cam_pose = np.eye(4)
        cam_pose[:3, 3] = [0.0, 1.3, 0.4]
        self.scene.add(camera, pose=cam_pose)
        
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        self.scene.add(light, pose=cam_pose)
        
        self.morph_nodes = []
        for node in self.scene.nodes:
            if node.mesh is not None:
                for prim in node.mesh.primitives:
                    if hasattr(prim, "targets") and prim.targets is not None:
                        num_shapes = int(prim.targets.positions.shape[0])
                        node.mesh.weights = np.zeros(num_shapes, dtype=np.float32)
                        mesh_name = node.name or node.mesh.name or "Unnamed_Mesh"
                        target_names = self.mesh_target_names.get(mesh_name)
                        self.morph_nodes.append((node, num_shapes, target_names))
                        break
        
        self.renderer = pyrender.OffscreenRenderer(w, h)

    def _build_mesh_target_name_map(self, gltf):
        mesh_target_names = {}
        if not gltf.model or not gltf.model.meshes: return mesh_target_names
        for mesh_def in gltf.model.meshes:
            name = getattr(mesh_def, "name", "Unnamed_Mesh")
            extras = getattr(mesh_def, "extras", {})
            if isinstance(extras, dict) and "targetNames" in extras:
                mesh_target_names[name] = extras["targetNames"]
        return mesh_target_names

    def render(self, blendshape_weights):
        model_weights = np.asarray(blendshape_weights, dtype=np.float32)
        metadata_weights = np.zeros(len(ARKIT_BLENDSHAPE_NAMES), dtype=np.float32)
        for beat_idx, metadata_idx in BEAT_TO_METADATA_IDX.items():
            if beat_idx < len(model_weights):
                metadata_weights[metadata_idx] = model_weights[beat_idx]

        for node, expected_length, target_names in self.morph_nodes:
            weights = np.zeros(expected_length, dtype=np.float32)
            if target_names:
                for t_idx, t_name in enumerate(target_names):
                    m_idx = self.metadata_name_to_idx.get(t_name)
                    if m_idx is not None and t_idx < expected_length:
                        weights[t_idx] = metadata_weights[m_idx]
            else:
                copy_len = min(expected_length, len(metadata_weights))
                weights[:copy_len] = metadata_weights[:copy_len]
            node.mesh.weights = weights

        color, _ = self.renderer.render(self.scene, flags=pyrender.RenderFlags.RGBA)
        return cv2.cvtColor(color, cv2.COLOR_RGBA2BGR)

class MicABSMonitor:
    def __init__(self, model_path, glb_path=None, log_callback=print, window_seconds=0.4, smoothing=0.35, top_k=12, input_device_index=None):
        self.log = log_callback
        self.RATE = 16000 # Engine might expect 48k or 16k; check your training. Using 16k for now.
        self.CHUNK = 1024
        self.smoothing = float(np.clip(smoothing, 0.0, 0.95))
        self.top_k = top_k
        self.input_device_index = input_device_index
        
        # Initialize Inference Engine (fast)
        self.engine = InferenceEngineFast(
            model_path=model_path,
            log_callback=log_callback,
            window_seconds=window_seconds,
            sample_rate=self.RATE,
        )

        self.p = pyaudio.PyAudio()
        self.stream = None
        self.is_running = False
        
        self.last_blendshape = np.zeros(52, dtype=np.float32)
        self.latest_blendshape = np.zeros(52, dtype=np.float32)
        self.last_rms = 0.0
        self.last_pred_ts = 0.0

        self.glb_engine = GLBEngine(Path(glb_path)) if glb_path else None

    def _calculate_rms(self, frame):
        data = np.frombuffer(frame, dtype=np.int16)
        if len(data) == 0: return 0.0
        return float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))

    def _stream_audio(self):
        try:
            self.stream = self.p.open(
                format=pyaudio.paInt16, channels=1, rate=self.RATE,
                input=True, frames_per_buffer=self.CHUNK,
                input_device_index=self.input_device_index
            )
            self.log("[ABS] Mic active. Feeding InferenceEngine...")

            while self.is_running:
                data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                self.last_rms = self._calculate_rms(data)
                
                # Push data into your custom engine
                self.engine.process_audio_chunk(data)
                
                # Retrieve processed weights
                weights = self.engine.get_latest_weights()
                if weights is not None:
                    # Apply smoothing
                    self.latest_blendshape = (self.smoothing * self.latest_blendshape + 
                                            (1.0 - self.smoothing) * weights)
                    self.last_pred_ts = time.time()

        except Exception as e:
            self.log(f"[ABS] Stream Error: {e}")

    def _render_monitor(self, blendshape):
        h, w = 760, 1700
        canvas = np.zeros((h, w, 3), dtype=np.uint8) + 25
        
        pred_age = time.time() - self.last_pred_ts
        live_color = (60, 210, 120) if pred_age < 0.25 else (40, 120, 220)
        cv2.putText(canvas, f"Engine Active - Latency: {pred_age:.3f}s", (30, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, live_color, 2)

        # Draw bars
        top_idx = np.argsort(-np.abs(blendshape))[:self.top_k]
        for i, idx in enumerate(top_idx):
            y = 100 + i * 45
            val = float(blendshape[idx])
            cv2.rectangle(canvas, (300, y), (300 + int(val * 400), y + 30), (120, 210, 255), -1)
            cv2.putText(canvas, ARKIT_BLENDSHAPE_NAMES[idx], (30, y + 22), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        if self.glb_engine:
            model_img = self.glb_engine.render(blendshape)
            canvas[20:740, 1100:1680] = model_img
        return canvas

    def start(self):
        self.is_running = True
        threading.Thread(target=self._stream_audio, daemon=True).start()
        
        while self.is_running:
            frame = self._render_monitor(self.latest_blendshape)
            cv2.imshow("InferenceEngine Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        self.stop()

    def stop(self):
        self.is_running = False
        if self.stream: self.stream.close()
        self.p.terminate()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--glb", default=None)
    args = parser.parse_args()

    idx = choose_audio_input_device(print)
    app = MicABSMonitor(
        model_path=args.model, 
        glb_path=args.glb, 
        input_device_index=idx
    )
    
    try:
        app.start()
    except KeyboardInterrupt:
        print("\n[ABS] Stopping...")
    finally:
        app.stop()