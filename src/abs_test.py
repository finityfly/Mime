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
# from transformers import Wav2Vec2Model


ARKIT_BLENDSHAPE_NAMES = [
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
    "noseSneerRight",
    "tongueOut",
]


class AudioToBlendshapeModel(torch.nn.Module):
    def __init__(self, num_blendshapes=52, hidden_dim=512, use_pretrained=False):
        super().__init__()
        self.use_pretrained = use_pretrained

        if self.use_pretrained:
            from transformers import Wav2Vec2Model
            self.audio_backbone = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
            self.audio_proj = torch.nn.Linear(768, hidden_dim)
        else:
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


class MicABSMonitor:
    def __init__(
        self,
        model_path,
        log_callback=print,
        sample_rate=16000,
        chunk_size=1024,
        window_seconds=0.4,
        smoothing=0.35,
        top_k=12,
    ):
        self.model_path = Path(model_path)
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
        self.bs_name_to_idx = {name: i for i, name in enumerate(ARKIT_BLENDSHAPE_NAMES)}

        self._load_model()

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

    def _build_mouth_rings_3d(self, ctrl, n=40):
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        width = 1.2 + 0.9 * ctrl["wide"] + 0.35 * ctrl["smile"] - 0.45 * ctrl["pucker"]
        height = 0.20 + 1.2 * ctrl["jaw"] + 0.45 * ctrl["lower"] - 0.35 * ctrl["close"]
        depth = 0.1 + 0.9 * ctrl["pucker"]

        x = width * np.cos(t)
        y = 0.9 * height * np.sin(t)
        z = 0.33 * depth * np.cos(2 * t)

        upper_mask = np.sin(t) > 0
        y[upper_mask] -= 0.15 * ctrl["upper"]
        y[~upper_mask] += 0.18 * ctrl["lower"]

        corner_pull = 0.23 * ctrl["smile"] * np.sign(np.cos(t))
        y += corner_pull * np.abs(np.cos(t))

        outer = np.stack([x, y, z], axis=1)

        inner_scale = 0.45 + 0.3 * ctrl["jaw"]
        inner = outer * np.array([inner_scale, inner_scale * 0.85, 0.75])
        inner[:, 2] -= 0.2 + 0.2 * ctrl["pucker"]
        return outer, inner

    def _project_points(self, pts, w, h, yaw=0.0, pitch=0.0):
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
        rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float32)

        pts_r = (pts @ ry.T) @ rx.T
        z = pts_r[:, 2] + 4.8
        f = min(w, h) * 0.43
        px = w * 0.5 + f * (pts_r[:, 0] / z)
        py = h * 0.54 + f * (pts_r[:, 1] / z)
        return np.stack([px, py], axis=1).astype(np.int32), pts_r

    def _render_mouth3d_panel(self, vals, w=600, h=720):
        ctrl = self._mouth_controls(vals)
        panel = np.zeros((h, w, 3), dtype=np.uint8)

        y_grad = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        panel[:, :, 0] = (18 + 22 * y_grad).astype(np.uint8)
        panel[:, :, 1] = (14 + 20 * y_grad).astype(np.uint8)
        panel[:, :, 2] = (24 + 18 * y_grad).astype(np.uint8)

        outer, inner = self._build_mouth_rings_3d(ctrl)
        yaw = (ctrl["smile_right"] - ctrl["smile_left"]) * 0.35
        pitch = (ctrl["jaw"] - 0.5 * ctrl["upper"]) * 0.22

        outer_2d, outer_3d = self._project_points(outer, w, h, yaw=yaw, pitch=pitch)
        inner_2d, inner_3d = self._project_points(inner, w, h, yaw=yaw, pitch=pitch)

        tris = []
        n = len(outer_2d)
        for i in range(n):
            j = (i + 1) % n
            t1 = np.array([outer_2d[i], outer_2d[j], inner_2d[i]], dtype=np.int32)
            t2 = np.array([inner_2d[i], outer_2d[j], inner_2d[j]], dtype=np.int32)
            z1 = float((outer_3d[i, 2] + outer_3d[j, 2] + inner_3d[i, 2]) / 3.0)
            z2 = float((inner_3d[i, 2] + outer_3d[j, 2] + inner_3d[j, 2]) / 3.0)
            tris.append((z1, t1))
            tris.append((z2, t2))

        tris.sort(key=lambda x: x[0], reverse=True)
        for z_avg, tri in tris:
            shade = np.clip((z_avg + 0.8) / 2.4, 0.0, 1.0)
            color = (
                int(40 + 65 * shade),
                int(25 + 45 * shade),
                int(130 + 95 * shade),
            )
            cv2.fillConvexPoly(panel, tri, color)

        cv2.polylines(panel, [outer_2d], True, (180, 220, 255), 2)
        cv2.polylines(panel, [inner_2d], True, (70, 95, 140), 2)

        cv2.putText(panel, "3D Virtual Mouth", (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (220, 236, 255), 2, cv2.LINE_AA)
        cv2.putText(
            panel,
            f"jaw:{ctrl['jaw']:.2f} smile:{ctrl['smile']:.2f} pucker:{ctrl['pucker']:.2f}",
            (22, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (205, 215, 230),
            1,
            cv2.LINE_AA,
        )
        return panel

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

        # Show top-K most active blendshapes so motion is easy to inspect quickly.
        vals = np.asarray(blendshape, dtype=np.float32)
        top_idx = np.argsort(-np.abs(vals))[: self.top_k]

        x0 = 26
        y0 = 176
        row_h = 44
        scale_px = 420
        center_x = x0 + 260

        cv2.line(canvas, (center_x, y0 - 20), (center_x, y0 + row_h * self.top_k + 6), (95, 95, 110), 1)

        for row, idx in enumerate(top_idx):
            y = y0 + row * row_h
            v = float(vals[idx])
            width = int(scale_px * min(1.0, abs(v)))
            color = (90, 220, 255) if v >= 0 else (95, 145, 255)

            cv2.rectangle(canvas, (center_x, y), (center_x + width, y + 26), color, -1)
            bs_name = ARKIT_BLENDSHAPE_NAMES[idx] if idx < len(ARKIT_BLENDSHAPE_NAMES) else f"bs[{idx:02d}]"
            cv2.putText(canvas, bs_name, (x0, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{v:.3f}", (center_x + width + 10, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (205, 205, 210), 1, cv2.LINE_AA)

        stats_y = y0 + row_h * self.top_k + 28
        cv2.putText(
            canvas,
            f"min={vals.min():.3f}  max={vals.max():.3f}  mean={vals.mean():.3f}  std={vals.std():.3f}",
            (x0, stats_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (210, 210, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(canvas, "Press q to quit", (x0, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (170, 190, 230), 1, cv2.LINE_AA)

        # ARKit mouth-specific debug strip based on canonical names.
        mouth_names = [
            "jawOpen",
            "mouthClose",
            "mouthFunnel",
            "mouthPucker",
            "mouthSmileLeft",
            "mouthSmileRight",
            "mouthStretchLeft",
            "mouthStretchRight",
            "mouthUpperUpLeft",
            "mouthUpperUpRight",
            "mouthLowerDownLeft",
            "mouthLowerDownRight",
        ]
        mx, my, mw = 560, 104, 280
        for i, name in enumerate(mouth_names):
            idx = self.bs_name_to_idx.get(name, None)
            value = float(np.clip(vals[idx], 0.0, 1.0)) if idx is not None and idx < len(vals) else 0.0
            y = my + i * 23
            cv2.putText(canvas, name, (mx, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (195, 205, 218), 1, cv2.LINE_AA)
            cv2.rectangle(canvas, (mx + 170, y), (mx + 170 + mw, y + 14), (55, 55, 60), -1)
            cv2.rectangle(canvas, (mx + 170, y), (mx + 170 + int(mw * value), y + 14), (110, 210, 255), -1)

        mouth_panel = self._render_mouth3d_panel(vals, w=580, h=720)
        x_panel, y_panel = 1088, 20
        canvas[y_panel : y_panel + mouth_panel.shape[0], x_panel : x_panel + mouth_panel.shape[1]] = mouth_panel

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
        cv2.destroyAllWindows()


def _cli_log(msg):
    print(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mic -> Audio2Blendshape realtime monitor")
    parser.add_argument("--model", required=True, help="Path to .pt checkpoint from ABS_train")
    parser.add_argument("--window-seconds", type=float, default=0.4, help="Audio window length fed into model")
    parser.add_argument("--smoothing", type=float, default=0.35, help="EMA smoothing factor for blendshape display")
    parser.add_argument("--top-k", type=int, default=12, help="Number of strongest blendshapes to display")
    args = parser.parse_args()

    app = MicABSMonitor(
        model_path=args.model,
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
