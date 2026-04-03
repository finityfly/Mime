# COMP 4107 Project - Group 10

Since we are using the industry-standard `pyproject.toml`, the "Getting Started" section is much cleaner. It focuses on environment synchronization rather than manual pip installs.

Here is a professional, speed-oriented **README.md** tailored for your project.

---

# AI Live Translator & 3D Overlay

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
git clone https://github.com/your-repo/project-group-10.git
cd project-group-10

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
INWORLD_AUTH_SIGNATURE=aHd......g==  # Use the Basic (Base64) field

# Hugging Face (Model Access)
HF_TOKEN=hf_your_token_here
```

### 4. Running the Pipeline
Activate the environment and run the central orchestrator:

```bash
# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Launch the live translation engine
python src/main.py
```

### 5. Hardware Calibration
If the STT is too sensitive or "hallucinating" words during silence:
1. Observe the **Peak Vol** logs in the console while silent.
2. Open `src/utils/stt_engine.py`.
3. Adjust `self.SILENCE_THRESHOLD` to be slightly above your ambient noise floor.