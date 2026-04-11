# Mime

<div align="center">

### Universal speech, visualized in real-time

</div>

> COMP 4107 Project - Group 10
> 
> Daniel Lu, Methira Herath

A high-performance, low-latency pipeline for real-time speech-to-speech translation with 3D facial animation support, designed for Windows/Zoom environments.

## Getting Started

### Project Structure

```
mime/
├── assets/
│   └── avatar.glb				 # Placeholder for scanned 3D mesh of user's face
├── data/                        # BEAT dataset (downloaded separately)
├── logs/                        # TensorBoard trial logs
├── models/
│   ├── best_fast.pt             # Best checkpoint — fast inference model
│   ├── best_slow.pt             # Best checkpoint — slow inference model
├── notebooks/
│   └── ABS_train.ipynb          # Lip-sync model training + grid search
├── reports/					 # Project proposal and final report
├── resources/
├── results/
├── src/
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── arkit_renderer.py    # Offscreen GLB rendering + morph targets
│   │   ├── inference_engine_fast.py   # Sliding-window real-time inference
│   │   ├── inference_engine_slow.py   # Higher-accuracy inference path
│   │   ├── media_bridge.py      # Virtual camera + virtual cable transport
│   │   ├── mt_engine.py         # LLaMA machine translation
│   │   ├── stt_engine.py        # Whisper speech-to-text
│   │   └── tts_engine.py        # Inworld text-to-speech
│   ├── abs_test.py              # ABS model smoke tests
│   ├── client_main.py           # Main entry point
│   ├── sts_main.py              # Speech-to-speech standalone runner
│   ├── video_client.py          # Video pipeline client
├── .env                         # API credentials
└── README.md
```

### 1. Prerequisites
* **Python 3.10+**
* **FFmpeg** (Required for audio processing)
* **NVIDIA GPU** (Optional, but highly recommended for 4-bit MT inference)

### 2. Environment Setup
We use `uv` for lightning-fast, reproducible dependency management. If you don't have it, install it via `pip install uv`.

```bash
# Clone the repository
git clone https://github.com/2026W-COMP4107/mime.git
cd mime

# Virtual environment setup
python -m venv .venv
source .venv/bin/activate

# If you don't have uv installed
pip install uv
# Create a virtual environment and install all dependencies from pyproject.toml
uv sync
```

#### Optional: Force CUDA PyTorch Versions with `uv`
If you need the exact CUDA-enabled PyTorch stack (for example `+cu118` wheels), this repo includes pinned versions in `pyproject.toml` and routes `torch`, `torchvision`, and `torchaudio` to the PyTorch CUDA index.

Use:

```bash
# Refresh lockfile using pinned versions
uv lock

# Install exactly what is locked
uv sync
```

If you hit this error on Windows during `uv sync`:
`triton==2.1.0 ... doesn't have a source distribution or wheel for the current platform`

regenerate the lockfile from the current `pyproject.toml` and sync again:

```bash
uv lock --refresh
uv sync
```

Quick GPU verification:

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

If you specifically want a requirements-style install equivalent to pip:

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu118 -r torch.txt
```

### 3. Configuration
Create a `.env` file in the root directory and populate it with your API credentials:

```ini
# Groq (ASR & MT)
GROQ_API_KEY=gsk_your_key_here

# Inworld (TTS)
INWORLD_API_KEY=aHd......g==  # Use the Basic (Base64) field

# Hugging Face (Model Access)
HF_TOKEN=hf_your_token_here
```

### 4. Run Zoom Mime Client
Activate the environment and launch the client:

```bash
# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Launch Zoom Mime Client (STT + MT + TTS + Zoom bridge)
python src/client_main.py --enable-zoom-bridge
```

Optional overrides:

```bash
python src/client_main.py --enable-zoom-bridge \
	--zoom-width 1280 \
	--zoom-height 720 \
	--zoom-fps 30 \
	--audio-rate 48000 \
	--audio-buffer 960 \
	--audio-device-name "CABLE Input"
```

## Zoom Mime Client Setup (Windows)

This section is a simple getting-started guide to route generated avatar video and TTS audio directly into Zoom.

### 1. Install required Windows apps
Install and verify these once on Windows:

1. **OBS Studio** (provides virtual camera backend commonly used by `pyvirtualcam` on Windows).
2. **VB-CABLE** (or VB-Audio virtual cable family).
3. **Zoom Desktop Client**.

After install:

1. Reboot Windows (important for audio device registration).
2. In Windows Sound settings, verify you can see a playback device containing one of:
	 - `CABLE Input`
	 - `VB-Audio`
3. In Zoom audio settings, verify you can select the corresponding cable microphone (often `CABLE Output`).

### 2. Zoom app configuration
In Zoom before joining a meeting:

1. **Video** -> Camera: select the virtual camera exposed by your system (`pyvirtualcam` backend).
2. **Audio** -> Microphone: select the cable microphone endpoint (often `CABLE Output`).
3. **Audio** -> disable auto volume if needed and tune manually to avoid pumping.

### 3. What starts when you run the client
When you launch `python src/client_main.py --enable-zoom-bridge`, the app starts:

1. Speech-to-Text (STT)
2. Machine Translation (MT)
3. Text-to-Speech (TTS)
4. Zoom bridge transport (virtual camera + virtual cable audio output)

The bridge auto-retries every 3 seconds if the camera or cable is temporarily unavailable.

### 4. Troubleshooting checklist

- **No camera in Zoom**:
	- Ensure OBS virtual camera support is installed.
	- Close apps that might lock the same virtual camera.
	- Look for bridge logs: `Virtual camera unavailable; retrying in 3s`.

- **No cable audio in Zoom**:
	- Confirm VB-CABLE is installed and visible in Windows Sound.
	- Launch with `--audio-device-name "CABLE Input"` to force device selection.

- **Choppy output under heavy inference**:
	- Lower `--zoom-fps` or output resolution.
	- Keep background GPU-heavy apps closed when possible.

- **Distortion/clipping**:
	- Keep source audio in range; the bridge already applies normalization and soft limiting.
	- If needed, reduce upstream TTS gain.

### 5. Downloading the Dataset Locally
To download the BEAT dataset into this repository's `data/` folder, install the Hugging Face CLI first, then run `hf download`.

Install Hugging Face CLI:

```bash
# macOS and Linux
curl -LsSf https://hf.co/cli/install.sh | bash
```

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"
```

Download dataset into the local `data` directory:

```bash
hf download H-Liu1997/BEAT --repo-type dataset --local-dir data
```

## Training the Lip-Sync Model

**1. Download BEAT dataset**

```bash
# macOS / Linux
curl -LsSf https://hf.co/cli/install.sh | bash

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"

# Download into data/
hf download H-Liu1997/BEAT --repo-type dataset --local-dir data
```

**2. Train**

```bash
uv add --dev ipykernel
jupyter notebook notebooks/ABS_train.ipynb
```

The notebook runs a grid search over learning rate, hidden dimension, and epoch budget. It supports two backbone modes:

- **Custom CNN** — trained from scratch on BEAT audio
- **Wav2Vec 2.0** — pretrained phonetic encoder for higher accuracy

**3. Monitor**

```bash
tensorboard --logdir=logs
```

Navigate to the **HParams** tab to compare trials. Best checkpoints are saved to `models/` as `.pt` files with their config dict embedded for reproducible loading.

### GPU Verification

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

For CUDA-specific wheels:

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu118 -r torch.txt
```
