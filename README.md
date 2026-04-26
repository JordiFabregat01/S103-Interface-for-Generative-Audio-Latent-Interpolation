# S103-Interface-for-Generative-Audio-Latent-Interpolation
This is the repository for the group S103 during the course of Taller de Musicologia

# SCAPES Project Setup & Installation Guide

This guide provides a robust installation process to avoid common issues with audio-processing libraries (like `llvmlite` and `librosa`) and ensures your Jupyter Notebooks can find the internal project modules regardless of where they are saved.

---

## 🛠 1. Environment Setup

To avoid compilation errors on macOS, we use **Conda** to provide pre-compiled binaries for complex C++ dependencies.

### Step 1: Create a Clean Environment
We recommend **Python 3.11**. It is currently the most stable version for audio machine learning packages.

```bash
# 1. Create a fresh environment
conda create -n TTM python=3.11 -y

# 2. Activate it
conda activate TTM

# 3. The "Conda-First" Install (Crucial)
conda install -c conda-forge llvmlite numba librosa -y

# 4. Install Remaining Requirements
uv pip install -r requirements.txt