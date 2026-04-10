import torch
import torch.nn.functional as F
import numpy as np
import math
from collections import deque
from queue import Queue, Empty

def lengths_to_padding_mask(lengths, max_len=None):
    if max_len is None:
        max_len = int(lengths.max().item())
    steps = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return steps >= lengths.unsqueeze(1)

class SinusoidalPositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        self.d_model = d_model
        self.register_buffer("pe", torch.empty(1, 0, d_model), persistent=False)
        self._extend_pe(max_len)

    def _extend_pe(self, max_len):
        if max_len <= self.pe.size(1):
            return
        pe = torch.zeros(max_len, self.d_model, device=self.pe.device)
        position = torch.arange(0, max_len, dtype=torch.float32, device=self.pe.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32, device=self.pe.device) * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        self._extend_pe(x.size(1))
        return x + self.pe[:, : x.size(1)].to(dtype=x.dtype, device=x.device)

class AudioToBlendshapeModel(torch.nn.Module):
    def __init__(self, num_blendshapes=52, hidden_dim=512, num_layers=4, num_heads=8, dropout=0.1, max_tokens=1200):
        super().__init__()
        self.max_tokens = int(max_tokens)
        self.frontend_specs = [(7, 2, 3), (5, 2, 2), (5, 2, 2)]
        
        self.audio_backbone = torch.nn.Sequential(
            torch.nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            torch.nn.BatchNorm1d(64), torch.nn.SiLU(),
            torch.nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2, bias=False),
            torch.nn.BatchNorm1d(128), torch.nn.SiLU(),
            torch.nn.Conv1d(128, hidden_dim, kernel_size=5, stride=2, padding=2, bias=False),
            torch.nn.BatchNorm1d(hidden_dim), torch.nn.SiLU(),
        )
        
        self.temporal_mixer = torch.nn.Sequential(
            torch.nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2, groups=hidden_dim, bias=False),
            torch.nn.BatchNorm1d(hidden_dim), torch.nn.SiLU(),
            torch.nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
        )

        self.input_norm = torch.nn.LayerNorm(hidden_dim)
        self.pos_enc = SinusoidalPositionalEncoding(hidden_dim)

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head_norm = torch.nn.LayerNorm(hidden_dim)

        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2), torch.nn.SiLU(),
            torch.nn.Dropout(dropout), torch.nn.Linear(hidden_dim // 2, num_blendshapes),
            torch.nn.Sigmoid(),
        )

    def forward(self, x, lengths=None):
        if x.dim() == 2: x = x.unsqueeze(1)
        y = self.audio_backbone(x)
        y = y + self.temporal_mixer(y)
        x = y.transpose(1, 2)

        # Basic inference assumes full length if not provided
        if lengths is None:
            out_lengths = torch.full((x.size(0),), x.size(1), dtype=torch.long, device=x.device)
        else:
            out_lengths = lengths 

        x = self.input_norm(x)
        x = self.pos_enc(x)
        padding_mask = lengths_to_padding_mask(out_lengths, x.size(1))
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        x = self.head_norm(x)
        return self.regressor(x), out_lengths

class InferenceEngineSlow:
    def __init__(self, model_path, log_callback=print, window_seconds=0.4, sample_rate=48000):
        self.log = log_callback
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_queue = Queue(maxsize=10)
        
        self.sample_rate = sample_rate
        self.window_size = int(sample_rate * window_seconds)
        self.audio_buffer = deque(maxlen=self.window_size)
        
        self.model = AudioToBlendshapeModel(
            hidden_dim=512,
            num_layers=4,
            num_heads=8
        )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()
        self.log(f"[Inference] Model loaded from {model_path} on {self.device}")

    def process_audio_chunk(self, audio_bytes: bytes):
        chunk = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        self.audio_buffer.extend(chunk.tolist())

        if len(self.audio_buffer) >= self.window_size:
            audio_tensor = torch.FloatTensor(list(self.audio_buffer)).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Forward returns (preds, lengths)
                preds, _ = self.model(audio_tensor)
                # Take the last frame of the sequence
                blendshapes = preds[0, -1].cpu().numpy()
            
            try:
                if self.output_queue.full(): self.output_queue.get_nowait()
                self.output_queue.put_nowait(blendshapes)
            except: pass

    def get_latest_weights(self):
        try: return self.output_queue.get_nowait()
        except Empty: return None