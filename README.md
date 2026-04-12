# Mime

<div align="center">

### Universal speech, visualized in real-time

</div>

A high-performance, low-latency pipeline for real-time speech-to-speech translation with 3D facial animation support, designed for Windows/Zoom environments.

---

## Getting Started

### 1. Prerequisites
* **Python 3.10+**
* **FFmpeg** (Required for audio processing)
* **Git LFS** (Required to pull large model weights and assets)
* **NVIDIA GPU** (Highly recommended for low-latency inference)
* **Windows OS** (Required for Zoom bridge functionality)

### 2. Environment Setup
We use `uv` for fast, reproducible dependency management.

```bash
# 1. Install Git LFS (if not already installed)
# Windows: download from git-lfs.github.com or 'git lfs install'
# macOS: brew install git-lfs
git lfs install

# 2. Clone the repository
git clone https://github.com/2026W-COMP4107/Mime.git
cd Mime

# 3. Pull Large File Storage (LFS) assets
git lfs pull

# 4. Setup virtual environment and dependencies
# If you don't have uv: pip install uv
uv sync
```

#### Optional: GPU & CUDA Optimization
If you need specific CUDA-enabled PyTorch wheels (e.g., `+cu118`), the `pyproject.toml` is pre-configured to route to the PyTorch CUDA index.

```bash
# Refresh and sync locked GPU dependencies
uv lock --refresh
uv sync

# Quick GPU verification
uv run python -c "import torch; print(f'Torch: {torch.__version__} | CUDA: {torch.version.cuda} | Available: {torch.cuda.is_available()}')"
```

### 3. Configuration
Create a `.env` file in the root directory:

```ini
# Groq (ASR & MT)
GROQ_API_KEY=gsk_your_key_here

# Inworld (TTS)
INWORLD_API_KEY=your_base64_key_here

# Hugging Face (Model Access)
HF_TOKEN=hf_your_token_here
```

---

## Project Structure

```text
mime/
├── assets/          # 3D meshes (avatar.glb) and LFS tracked assets
├── data/            # BEAT dataset (downloaded separately)
├── models/          # Trained checkpoints (.pt files)
├── notebooks/       # Lip-sync training & grid search (ABS_train.ipynb)
├── reports/         # Proposal and final documentation
├── src/
│   ├── utils/       # Engines: ASR (Whisper), MT (LLaMA), TTS (Inworld)
│   ├── client_main.py # Main entry point
│   └── sts_main.py   # Standalone STS runner
└── .env             # Local environment secrets
```

---

## Zoom Mime Client (Windows)

This bridge routes generated avatar video and TTS audio directly into Zoom via a virtual camera and cable.

### 1. Windows Dependencies
1. **OBS Studio**: Provides the virtual camera backend.
2. **VB-CABLE**: Virtual audio cable for routing TTS to Zoom input.
3. **Zoom Desktop Client**.

### 2. Launching the Pipeline
```bash
# Activate environment
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Launch with Zoom bridge enabled
python src/client_main.py --enable-zoom-bridge --audio-device-name "CABLE Input"
```

### 3. Zoom Configuration
* **Video**: Select the OBS Virtual Camera (via `pyvirtualcam`).
* **Audio**: Select `CABLE Output` (VB-Audio) as your Microphone.

---

## Training & Dataset

### 1. Download BEAT Dataset
```bash
# Install HF CLI
# Windows (PS): powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"
hf download H-Liu1997/BEAT --repo-type dataset --local-dir data
```

### 2. Train Lip-Sync Model
```bash
uv add --dev ipykernel
jupyter notebook notebooks/ABS_train.ipynb
```
The notebook supports **Custom CNN** or **Wav2Vec 2.0** backbones. Monitor progress via TensorBoard:
```bash
tensorboard --logdir=logs
```

---

## Troubleshooting
* **Missing Models**: Ensure you ran `git lfs pull` after cloning.
* **Triton Errors (Windows)**: Run `uv lock --refresh` then `uv sync` to fix platform-specific wheel issues.
* **No Video in Zoom**: Ensure OBS is installed and no other app is locking the virtual camera device.