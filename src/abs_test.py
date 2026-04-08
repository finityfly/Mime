import argparse
from platform import node
import queue
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pyaudio
from pyrender import primitive
import torch
import pyrender
import os
import gltflib

np.product = np.prod 

# Disable PyOpenGL error checking for better performance
os.environ ["PYOPENGL_ERROR_CHECKING"] = "0"  

# order of megumin dataset
ARKIT_BLENDSHAPE_NAMES = ['eyeBlinkLeft', 
                          'eyeLookDownLeft', 
                          'eyeLookInLeft', 
                          'eyeLookOutLeft', 
                          'eyeLookUpLeft', 
                          'eyeSquintLeft', 
                          'eyeWideLeft', 
                          'eyeBlinkRight', 
                          'eyeLookDownRight', 
                          'eyeLookInRight', 
                          'eyeLookOutRight', 
                          'eyeLookUpRight', 
                          'eyeSquintRight', 
                          'eyeWideRight', 
                          'jawForward', 
                          'jawLeft', 
                          'jawRight', 
                          'jawOpen', 
                          'mouthClose', 
                          'mouthFunnel', 
                          'mouthPucker', 
                          'mouthRight', 
                          'mouthLeft', 
                          'mouthSmileLeft', 
                          'mouthSmileRight', 
                          'mouthFrownRight', 
                          'mouthFrownLeft', 
                          'mouthDimpleLeft', 
                          'mouthDimpleRight', 
                          'mouthStretchLeft', 
                          'mouthStretchRight', 
                          'mouthRollLower', 
                          'mouthRollUpper', 
                          'mouthShrugLower', 
                          'mouthShrugUpper', 
                          'mouthPressLeft', 
                          'mouthPressRight', 
                          'mouthLowerDownLeft', 
                          'mouthLowerDownRight', 
                          'mouthUpperUpLeft', 
                          'mouthUpperUpRight', 
                          'browDownLeft', 
                          'browDownRight', 
                          'browInnerUp', 
                          'browOuterUpLeft', 
                          'browOuterUpRight', 
                          'cheekPuff', 
                          'cheekSquintLeft', 
                          'cheekSquintRight', 
                          'noseSneerLeft', 
                          'noseSneerRight', 
                          'tongueOut']

# BEAT training order from dataset JSON `names` field (51 channels).
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

# Explicit mapping requested: BEAT output index -> metadata index.
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
        self.w = w
        self.h = h
        self.log = print
        self.model_name_to_idx = {
            name: i for i, name in enumerate(MODEL_OUTPUT_BLENDSHAPE_NAMES)
        }
        self.metadata_name_to_idx = {
            name: i for i, name in enumerate(ARKIT_BLENDSHAPE_NAMES)
        }
        
        self.log(f"[GLB] Loading {glb_path}...")
        gltf = gltflib.GLTF.load(str(glb_path))
        self.scene = pyrender.Scene.from_gltflib_scene(gltf)
        self.log("[GLB] Loaded via gltflib -> pyrender (morph targets preserved).")
        self.mesh_target_names = self._build_mesh_target_name_map(gltf)
        
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=w/h)
        cam_pose = np.array([
            [1.0,  0.0,  0.0,  0.0],
            [0.0,  1.0,  0.0,  1.3], 
            [0.0,  0.0,  1.0,  0.4],
            [0.0,  0.0,  0.0,  1.0]
        ])
        self.scene.add(camera, pose=cam_pose)
        
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        self.scene.add(light, pose=cam_pose)
        
        # --- THE MULTI-MESH FIX ---
        self.morph_nodes = []
        for node in self.scene.nodes:
            if node.mesh is not None:
                for prim in node.mesh.primitives:
                    has_targets = hasattr(prim, "targets") and prim.targets is not None
                    if has_targets and hasattr(prim.targets, "positions") and prim.targets.positions is not None:
                        num_blendshapes = int(prim.targets.positions.shape[0])
                        node.mesh.weights = np.zeros(num_blendshapes, dtype=np.float32)
                        
                        # Grab the name so we can see exactly what we are animating
                        mesh_name = node.name or node.mesh.name or "Unnamed_Mesh"
                        target_names = self.mesh_target_names.get(mesh_name)
                        if target_names and len(target_names) != num_blendshapes:
                            self.log(
                                f"[GLB] Warning: '{mesh_name}' targetNames count ({len(target_names)}) "
                                f"!= target count ({num_blendshapes}); falling back to index mapping."
                            )
                            target_names = None

                        self.morph_nodes.append((node, num_blendshapes, target_names))
                        
                        if target_names:
                            self.log(f"[GLB] Success: Hooked up '{mesh_name}' ({num_blendshapes} shapes, named mapping)")
                        else:
                            self.log(f"[GLB] Success: Hooked up '{mesh_name}' ({num_blendshapes} shapes, index mapping)")
                        break # Move to the next node
        
        if not self.morph_nodes:
            self.log("[GLB] ERROR: Absolutely zero blendshapes found in this file.")
            
        self.renderer = pyrender.OffscreenRenderer(w, h)

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
                self.log(f"[GLB] targetNames for '{mesh_name}': {target_names}")
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
            if beat_idx < len(model_weights) and metadata_idx < len(metadata_weights):
                metadata_weights[metadata_idx] = float(model_weights[beat_idx])

        for node, expected_length, target_names in self.morph_nodes:
            try:
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
            except Exception as e:
                self.log(f"[GLB Render Error] {e}")

        color, _ = self.renderer.render(self.scene, flags=pyrender.RenderFlags.RGBA)
        return cv2.cvtColor(color, cv2.COLOR_RGBA2BGR)

class MicABSMonitor:
    def __init__(
        self,
        model_path,
        glb_path=None,
        log_callback=print,
        sample_rate=16000,
        chunk_size=1024,
        window_seconds=0.4,
        smoothing=0.35,
        top_k=12,
    ):
        self.model_path = Path(model_path)
        self.glb_path = Path(glb_path)
        self.log = log_callback

        self.RATE = sample_rate
        self.CHUNK = chunk_size
        self.CHANNELS = 1
        self.FORMAT = pyaudio.paInt16

        self.window_chunks = max(3, int((self.RATE * window_seconds) / self.CHUNK))
        self.smoothing = float(np.clip(smoothing, 0.0, 0.95))
        self.top_k = max(1, min(52, int(top_k)))

        self.p = pyaudio.PyAudio()
        self.stream = None

        self.audio_queue = queue.Queue(maxsize=2)
        self.pred_queue = queue.Queue(maxsize=2)
        self.is_running = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._configure_torch_backends()
        self.model = None
        self.last_blendshape = np.zeros(52, dtype=np.float32)
        self.latest_blendshape = np.zeros(52, dtype=np.float32)
        self.last_rms = 0.0
        self.last_pred_ts = 0.0
        self.bs_name_to_idx = {name: i for i, name in enumerate(MODEL_OUTPUT_BLENDSHAPE_NAMES)}

        self._load_model()

        self.glb_engine = None
        if self.glb_path and self.glb_path.exists():
            self.glb_engine = GLBEngine(self.glb_path, w=580, h=720)

    def _configure_torch_backends(self):
        # Some CPUs cannot initialize NNPACK; disable it to avoid repeated warnings.
        if self.device.type != "cpu":
            return

        nnpack_backend = getattr(torch.backends, "nnpack", None)
        if nnpack_backend is None:
            return

        try:
            nnpack_backend.enabled = False
            self.log("[ABS] Disabled NNPACK backend for CPU compatibility.")
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
            f"[ABS] Loaded model: {self.model_path.name} "
            f"(hidden_dim={hidden_dim}, use_pretrained={use_pretrained}, device={self.device})"
        )

    def _calculate_rms(self, frame):
        data = np.frombuffer(frame, dtype=np.int16)
        if len(data) == 0:
            return 0.0
        return float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))

    def _stream_audio(self):
        try:
            self.stream = self.p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK,
            )
            self.log("[ABS] Mic active. Streaming audio into model...")

            rolling = deque(maxlen=self.window_chunks)

            while self.is_running:
                data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                rms = self._calculate_rms(data)
                self.last_rms = rms

                rolling.append(data)

                if len(rolling) >= self.window_chunks:
                    try:
                        if self.audio_queue.full():
                            _ = self.audio_queue.get_nowait()
                        self.audio_queue.put_nowait(list(rolling))
                    except queue.Empty:
                        pass
                    except queue.Full:
                        pass
        except Exception as e:
            self.log(f"[ABS] Hardware Error: {e}")
        finally:
            if self.stream is not None:
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                except Exception:
                    pass
                self.stream = None

    def _frames_to_audio(self, frames, gain=7.0):
        audio_int16 = np.frombuffer(b"".join(frames), dtype=np.int16)
        if len(audio_int16) == 0:
            return None
        
        # Convert to float and apply gain
        audio_float = audio_int16.astype(np.float32) / 32768.0
        audio_float = audio_float * gain 
        
        # Hard clip to prevent blowing out the tensor (-1.0 to 1.0)
        audio = np.clip(audio_float, -1.0, 1.0)
        
        return torch.from_numpy(audio).unsqueeze(0)

    def _predict_blendshape(self, audio_tensor):
        with torch.no_grad():
            model_in = audio_tensor.to(self.device)
            out = self.model(model_in)
            blendshape = out[0, -1].detach().cpu().numpy().astype(np.float32)

        # Exponential smoothing to reduce jitter without freezing movement.
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
                self.latest_blendshape = blendshape
                self.last_pred_ts = time.time()

                if self.pred_queue.full():
                    _ = self.pred_queue.get_nowait()
                self.pred_queue.put_nowait(blendshape)
            except Exception as e:
                self.log(f"[ABS] Inference Error: {e}")
    
    def _render_monitor(self, blendshape):
        h, w = 760, 1700
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[:] = (22, 22, 28)

        cv2.putText(canvas, "Mic -> Audio2Blendshape Monitor", (26, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (220, 235, 255), 2, cv2.LINE_AA)

        pred_age = max(0.0, time.time() - self.last_pred_ts)
        live_color = (60, 210, 120) if pred_age < 0.25 else (40, 120, 220)
        cv2.circle(canvas, (30, 76), 8, live_color, -1)
        cv2.putText(canvas, f"pred_age={pred_age:.3f}s", (48, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1, cv2.LINE_AA)

        rms_norm = min(1.0, self.last_rms / 1200.0)
        bar_left, bar_top, bar_w, bar_h = 26, 104, 280, 20
        cv2.rectangle(canvas, (bar_left, bar_top), (bar_left + bar_w, bar_top + bar_h), (55, 55, 60), -1)
        cv2.rectangle(canvas, (bar_left, bar_top), (bar_left + int(bar_w * rms_norm), bar_top + bar_h), (120, 210, 255), -1)
        cv2.putText(canvas, f"mic_rms={self.last_rms:.1f}", (bar_left, bar_top + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        vals = np.asarray(blendshape, dtype=np.float32)
        top_idx = np.argsort(-np.abs(vals))[: self.top_k]

        x0, y0, row_h, scale_px, center_x = 26, 176, 44, 420, 286
        cv2.line(canvas, (center_x, y0 - 20), (center_x, y0 + row_h * self.top_k + 6), (95, 95, 110), 1)

        for row, idx in enumerate(top_idx):
            y = y0 + row * row_h
            v = float(vals[idx])
            width = int(scale_px * min(1.0, abs(v)))
            color = (90, 220, 255) if v >= 0 else (95, 145, 255)

            cv2.rectangle(canvas, (center_x, y), (center_x + width, y + 26), color, -1)
            bs_name = ARKIT_BLENDSHAPE_NAMES[idx] if idx < len(ARKIT_BLENDSHAPE_NAMES) else f"bs[{idx:02d}]"
            cv2.putText(canvas, bs_name, (x0, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

        # Decide whether to render the 3D GLB or just a blank placeholder
        x_panel, y_panel = 1088, 20
        if self.glb_engine:
            try:
                # Render the GLB
                model_render = self.glb_engine.render(vals)
                canvas[y_panel : y_panel + 720, x_panel : x_panel + 580] = model_render
            except Exception as e:
                cv2.putText(canvas, "GLB Render Error", (x_panel + 50, y_panel + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(canvas, "Pass --glb to see 3D model", (x_panel + 100, y_panel + 360), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)

        return canvas

    def _visualize_loop(self):
        while self.is_running:
            try:
                while True:
                    self.latest_blendshape = self.pred_queue.get_nowait()
            except queue.Empty:
                pass

            frame = self._render_monitor(self.latest_blendshape)
            cv2.imshow("ABS Blendshape Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.stop()
                break

    def start(self):
        if self.is_running:
            return

        self.is_running = True
        threading.Thread(target=self._stream_audio, daemon=True).start()
        threading.Thread(target=self._inference_loop, daemon=True).start()
        self._visualize_loop()

    def stop(self):
        self.is_running = False
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        try:
            self.p.terminate()
        except Exception:
            pass
        cv2.destroyAllWindows()

def _cli_log(msg):
    print(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mic -> Audio2Blendshape realtime monitor")
    parser.add_argument("--model", required=True, help="Path to .pt checkpoint from ABS_train")
    parser.add_argument("--glb", type=str, default=None, help="Path to .glb file to render")
    parser.add_argument("--window-seconds", type=float, default=0.4, help="Audio window length fed into model")
    parser.add_argument("--smoothing", type=float, default=0.35, help="EMA smoothing factor for blendshape display")
    parser.add_argument("--top-k", type=int, default=12, help="Number of strongest blendshapes to display")
    args = parser.parse_args()

    app = MicABSMonitor(
        model_path=args.model,
        glb_path=args.glb,
        log_callback=_cli_log,
        window_seconds=args.window_seconds,
        smoothing=args.smoothing,
        top_k=args.top_k,
    )
    try:
        app.start()
    except KeyboardInterrupt:
        _cli_log("[ABS] Interrupted by user.")
    finally:
        app.stop()
