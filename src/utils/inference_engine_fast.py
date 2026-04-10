import torch
import numpy as np
from collections import deque
from queue import Queue, Empty


class AudioToBlendshapeModel(torch.nn.Module):
    # Minimal model matching `abs_test.py`
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

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(256, num_blendshapes),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.audio_backbone(x).transpose(1, 2)

        x = self.transformer(x)
        return self.regressor(x)


class InferenceEngineFast:
    def __init__(self, model_path, log_callback=print, window_seconds=0.4, sample_rate=48000):
        self.log = log_callback
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_queue = Queue(maxsize=10)

        self.sample_rate = sample_rate
        self.window_size = int(sample_rate * window_seconds)
        self.audio_buffer = deque(maxlen=self.window_size)

        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint.get("config", {})
        hidden_dim = config.get("hidden_dim", 512)
        use_pretrained = config.get("use_pretrained", False)

        self.model = AudioToBlendshapeModel(num_blendshapes=52, hidden_dim=hidden_dim, use_pretrained=use_pretrained).to(self.device)

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.log(f"[Inference] Model loaded from {model_path} on {self.device} (hidden_dim={hidden_dim}, use_pretrained={use_pretrained})")

    def process_audio_chunk(self, audio_bytes: bytes):
        chunk = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        self.audio_buffer.extend(chunk.tolist())

        if len(self.audio_buffer) >= self.window_size:
            audio_tensor = torch.FloatTensor(list(self.audio_buffer)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self.model(audio_tensor)
                preds = out[0] if isinstance(out, (tuple, list)) else out
                blendshapes = preds[0, -1].cpu().numpy()

            try:
                if self.output_queue.full():
                    self.output_queue.get_nowait()
                self.output_queue.put_nowait(blendshapes)
            except Exception:
                pass

    def get_latest_weights(self):
        try:
            return self.output_queue.get_nowait()
        except Empty:
            return None
