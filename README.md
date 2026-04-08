# Mime

<div align="center">

### Universal speech, visualized in real-time

</div>

> COMP 4107 Project - Group 10
> 
> Daniel Lu, Methira Herath

A high-performance, low-latency pipeline for real-time speech-to-speech translation with 3D facial animation support, designed for Windows/Zoom environments.

## Getting Started

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

### 3. Configuration
Create a `.env` file in the root directory and populate it with your API credentials:

```ini
# Groq (ASR & MT)
GROQ_API_KEY=gsk_your_key_here

# Inworld (TTS)
INWORLD_AUTH_SIGNATURE=aHd......g==  # Base64 field

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
hf download H-Liu1997/BEAT --repo-type dataset --local-dir data/beat_english_v0.2.1
```

## Training the Lip-Sync Model

The Visual Lip-Sync Pipeline uses a Transformer-based model to map audio features to 52 ARKit blendshape coefficients. We use the **BEAT Dataset** for high-fidelity training.

### 1. Training Environment
The training is conducted via the `notebooks/train.ipynb` file. This notebook is designed to run in a Jupyter environment with a GPU.

```bash
# Ensure dev dependencies are installed (including ipykernel)
uv add --dev ipykernel

# Launch Jupyter
jupyter notebook notebooks/ABS_train.ipynb
```

### 2. Hyperparameter Fitting
The training script implements a grid-search approach to find the optimal model. You can toggle between:
* **Custom Phonetic-CNN:** A from-scratch encoder that learns phonetics specifically for this task.
* **Pre-trained Wav2Vec 2.0:** A robust backbone that provides higher accuracy but slightly more latency.

### 3. Monitoring with TensorBoard
All trials log their loss curves and hyperparameters to the `logs/` directory in the project root. To visualize the "fitting" process:

```bash
# From the project root
tensorboard --logdir=logs
```
Navigate to the **HParams** tab in the browser to compare different model configurations.

### 4. Model Artifacts
The training loop automatically performs the following:
1. **Validation:** Checks performance against a 10% hold-out set of the BEAT dataset.
2. **Serialization:** Saves the best-performing version of each trial as a `.pt` file in the `models/` directory.
3. **Traceability:** Each saved model includes its specific configuration dictionary, making it easy to load into the live orchestrator.
