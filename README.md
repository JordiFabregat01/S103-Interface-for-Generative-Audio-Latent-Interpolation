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

## Option A — Using `uv` (Recommended)

Install `uv` if needed:

```bash
pip install uv
```

Install dependencies:

```bash
# Main backend dependencies
uv pip install -r requirements.txt

# SCAPES dependencies
uv pip install -r modules/scapes/requirements.txt
```

---

## Option B — Using `pip`

```bash
pip install -r requirements.txt
pip install -r modules/scapes/requirements.txt
```

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

Now install the rest using `uv` or `pip`:

```bash
uv pip install -r requirements.txt
uv pip install -r modules/scapes/requirements.txt
```

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
