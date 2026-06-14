# S103 – Interface for Generative Audio Latent Interpolation

This repository contains the work of group **S103** for the *Taller de Musicologia* course. It provides an interface for exploring generative audio through latent space interpolation, combining a Python backend with a TypeScript/Vite frontend, and integrating the **SCAPES** module.

---

# 📦 Project Structure

* **Backend**: Python (FastAPI + audio/ML dependencies)
* **Frontend**: Vite + TypeScript
* **Submodule**: SCAPES (audio processing and synthesis)

---

# 🚀 Quickstart

## 1. Clone the Repository (with Submodules)

This project depends on the SCAPES submodule. Clone everything at once:

```bash
git clone --recurse-submodules https://github.com/your-repo/S103-Interface-for-Generative-Audio-Latent-Interpolation.git
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

---

## 2. Installation Overview

The project has two main parts:

* **Backend (Python)**
* **Frontend (Node.js)**

You can install them independently, but both are required for full functionality.

---

# 🐍 Backend Installation

Create a virtual environment with the standard-library `venv` module and install
the pinned dependencies. `requirements.txt` is **complete** — it already bundles
the SCAPES, CLAP/`msclap`, and backend dependencies, so it's the only file you
need.

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires **Python 3.12**. `requirements.txt` pins the **CUDA 12.8 (cu128)** torch
build for GPU; see the comment at the top of the file to switch to a CPU-only
build.

---

# ⚠️ macOS Special Setup (Recommended)

Some audio and ML dependencies (such as `llvmlite`, `numba`, and `librosa`) can fail to compile on macOS when installed via pip.

To avoid these issues, **use Conda to pre-install critical packages**.

## Step-by-Step (macOS)

### 1. Create a Clean Environment

We recommend Python 3.11 for best compatibility:

```bash
conda create -n TTM python=3.11 -y
conda activate TTM
```

### 2. Install Critical Dependencies via Conda

This step avoids compilation errors:

```bash
conda install -c conda-forge llvmlite numba librosa -y
```

### 3. Install Remaining Dependencies

Now install the rest with `pip` (inside the conda env):

```bash
pip install -r requirements.txt
```

> On macOS there is no CUDA: edit the two `torch`/`torchaudio` lines in
> `requirements.txt` to the plain `==2.11.0` (CPU) versions and drop the
> `--extra-index-url`, per the note at the top of the file.

---

## 💡 Why this matters

* `llvmlite` and `numba` rely on compiled C/C++ code
* macOS often lacks compatible toolchains by default
* Conda provides pre-built binaries, avoiding build failures

---
# ⚡ CUDA Setup & Verification Guide

This short guide explains how to install CUDA support and verify that it is working correctly on your system.

---

## 🧠 Requirements

* NVIDIA GPU (CUDA **does NOT work** on macOS with Apple Silicon or AMD GPUs)
* Compatible NVIDIA drivers installed
* Python environment (recommended: Python 3.10–3.11)

---

## 🔧 1. Install CUDA Support (PyTorch)

The easiest way to get CUDA working for this project is through PyTorch.

Install PyTorch with CUDA support:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

> ℹ️ Replace `cu121` with the version compatible with your system if needed.

---

## 🔍 2. Verify CUDA at System Level

Check that your GPU and drivers are correctly installed:

```bash
nvidia-smi
```

✅ Expected:

* GPU name appears
* Driver version shown
* CUDA version listed

---

## 🧪 3. Verify CUDA in Python

Open a Python terminal:

```bash
python
```

Run the following:

```python
import torch

print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
```


---

# 🌐 Frontend Installation

Navigate to the frontend directory:

```bash
cd app/frontend
```

Install dependencies:

```bash
npm install
```

---

# ▶️ Running the Project

## Backend

```bash
cd app/backend
python -m uvicorn main:app --reload
```

API will be available at:

```
http://localhost:8000
```

---

## Frontend

```bash
cd app/frontend
npm run dev
```

Frontend will run at:

```
http://localhost:5173
```

*(Port may change if already in use)*

---

# 🧪 Verification

If everything is correctly installed:

* Backend should start without import errors
* Frontend should load in the browser
* API requests should connect successfully

---

# 🧪 Testing — Model Evaluation

The trained model can be evaluated against the source dataset with three
audio-distribution distances (lower is better):

* **FAD-VGGish** — Fréchet distance over VGGish embeddings
* **KAD-CLAP** — Kernel (MMD) distance over CLAP-2023 embeddings
* **KAD-TexStat** — Kernel distance over TexStat embeddings

For every source sound the script resynthesizes ~60s of the same texture, then
compares the embedding distributions of the originals vs the generations and
writes the results to an Excel file.

The evaluation runs in **two phases** that talk only through WAV files on disk,
because they need **different, incompatible environments**:

* `generate` → runs in the **project venv** (torch 2.11 / Python 3.12 / msclap /
  SCAPES) and writes `reference/` + `generated/` WAVs
* `score` → runs in a **separate Python 3.11 venv** for `kadtk` (which pins
  `torch<2.6` and `python<3.12`, so it must NOT share the project env) and reads
  those WAVs to produce the Excel

> ⚠️ Run all commands below from `app/backend`.

### 1. One-time setup for the scoring venv

`KAD-TexStat` uses the `ddsp_textures` submodule (and its `torch_filterbanks`),
so initialise submodules, then build the isolated env. Run from the **repo root**:

**Windows (PowerShell):**

```powershell
# submodules: ddsp_textures + scapes
git submodule update --init modules/scapes modules/ddsp_textures

# ddsp_textures references torch_filterbanks as a bare gitlink (no upstream
# .gitmodules entry), so clone it explicitly into place:
git clone https://github.com/cordutie/torch_filterbanks.git `
  modules/ddsp_textures/experiments/texstat/torch_filterbanks

# dedicated Python 3.11 env for scoring (kept separate from the project venv)
py -3.11 -m venv .venv-eval
.venv-eval\Scripts\python -m pip install -r app/backend/inference/evaluation/requirements-scoring.txt
```

**macOS / Linux (bash):**

```bash
git submodule update --init modules/scapes modules/ddsp_textures
git clone https://github.com/cordutie/torch_filterbanks.git \
  modules/ddsp_textures/experiments/texstat/torch_filterbanks
python3.11 -m venv .venv-eval
.venv-eval/bin/python -m pip install -r app/backend/inference/evaluation/requirements-scoring.txt
```

> ℹ️ `torchaudio` must match the `torch` version `kadtk` resolves (e.g. torch
> 2.5.x → torchaudio 2.5.x). If the install picks an incompatible build, run
> `.venv-eval/bin/python -m pip install "torchaudio==2.5.*"` afterwards.

### 2. Smoke Test (single sound)

Quickly checks the full pipeline end-to-end on one sound. Both phases need
**`CWD = app/backend`**.

**Windows (PowerShell):**

```powershell
# PHASE 1 — generate (PROJECT venv)
cd app/backend
..\..\.venv\Scripts\python -m inference.evaluation.evaluate_model --phase generate --limit 1
cd ../..

# PHASE 2 — score (SCORING venv)
.venv-eval\Scripts\Activate.ps1
cd app/backend
python -m inference.evaluation.evaluate_model --phase score
cd ../.. ; deactivate
```

**macOS / Linux (bash):**

```bash
# PHASE 1 — generate (PROJECT venv)
(cd app/backend && ../../.venv/bin/python -m inference.evaluation.evaluate_model --phase generate --limit 1)

# PHASE 2 — score (SCORING venv)
source .venv-eval/bin/activate
(cd app/backend && python -m inference.evaluation.evaluate_model --phase score)
deactivate
```

Confirm that `app/backend/inference/evaluation/out/reference/` and `.../generated/`
each contain one ~60s WAV, and that `.../out/eval_results.xlsx` opens with finite
**FAD-VGGish / KAD-CLAP / KAD-TexStat** values.

### 3. Complete Test (full dataset)

Once the smoke test passes, run the same two phases without `--limit`:

**Windows (PowerShell):**

```powershell
# PHASE 1 — generate all sounds (PROJECT venv)
cd app/backend
..\..\.venv\Scripts\python -m inference.evaluation.evaluate_model --phase generate
cd ../..

# PHASE 2 — score (SCORING venv)
.venv-eval\Scripts\Activate.ps1
cd app/backend
python -m inference.evaluation.evaluate_model --phase score
cd ../.. ; deactivate
```

**macOS / Linux (bash):**

```bash
(cd app/backend && ../../.venv/bin/python -m inference.evaluation.evaluate_model --phase generate)
source .venv-eval/bin/activate
(cd app/backend && python -m inference.evaluation.evaluate_model --phase score)
deactivate
```

The Excel (`app/backend/inference/evaluation/out/eval_results.xlsx`) has one row
per sound plus a pooled **OVERALL** row — the pooled row is the headline number.
Scores reflect the full dataset (no cherry-picking); selecting the best
generations would lower the distances.

> 💡 Useful flags: `--data-dir <dir>` (evaluate a different WAV folder),
> `--duration`, `--nfe`, `--seed`, `--only <stem>`, `--limit N`, `--device cpu|cuda`.
> See `--help` for the full list.

### What to expect on the first run

* The scoring venv **downloads model weights** (VGGish, CLAP) on first use.
* The `generate` phase loads the ~887 MB SCAPES flow model and resynthesizes
  ~60s per sound, so the full dataset takes a while (GPU strongly recommended).
* The two phases are decoupled through the WAV files on disk, so you can re-run
  `--phase score` as often as you like without regenerating audio.

---

# 🧩 Working with SCAPES

The SCAPES module is included as a submodule:

* Located in: `modules/scapes`
* Has its own `requirements.txt`
* Must be installed separately (already covered above)

If you encounter issues:

```bash
git submodule update --init --recursive
```

---

# 🛠 Troubleshooting

### Common Issues

**1. Module not found errors**

* Ensure both `requirements.txt` files are installed

**2. Audio libraries failing to install (macOS)**

* Use the Conda setup above

**3. Submodule missing files**

* Re-run:

  ```bash
  git submodule update --init --recursive
  ```

**4. Port already in use**

* Backend: change uvicorn port
* Frontend: Vite will auto-suggest another port

---

# 📌 Notes

* Python 3.11 is strongly recommended
* Node.js ≥ 18 recommended
* Conda is optional but highly recommended on macOS

---

# 👥 Authors

Group **S103** – Taller de Musicologia

---

# 📄 License

(Add your license here)

---