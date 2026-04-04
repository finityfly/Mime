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
INWORLD_AUTH_SIGNATURE=aHd......g==  # Base64 field

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
