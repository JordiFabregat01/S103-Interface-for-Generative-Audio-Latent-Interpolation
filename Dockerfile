# syntax=docker/dockerfile:1.6
#
# Backend container for the S103 latent-interpolation interface.
# Built for Hugging Face Spaces (Docker SDK): listens on port 7860, runs as
# UID 1000, no GPU required. To run locally:
#
#     docker build -t s103-backend .
#     docker run --rm -p 7860:7860 s103-backend
#

FROM python:3.12-slim AS base

# ---- System dependencies --------------------------------------------------
# ffmpeg + libsndfile: audio I/O for librosa/soundfile.
# git: in case any pip dep installs from a git URL.
# build-essential: fallback for any wheel that needs compiling.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---- Non-root user (Hugging Face Spaces convention) -----------------------
RUN useradd --create-home --shell /bin/bash --uid 1000 user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/user/.cache/huggingface

WORKDIR /home/user/app
USER user

# ---- Python dependencies --------------------------------------------------
# Install the CPU-only torch wheel first. Doing this before requirements.txt
# prevents pip from resolving torch via PyPI and pulling the ~2 GB CUDA build,
# which would blow past the 50 GB Space limit and slow cold starts.
RUN pip install --no-cache-dir --user --upgrade pip \
 && pip install --no-cache-dir --user \
        --index-url https://download.pytorch.org/whl/cpu \
        torch torchaudio

# Project dependencies live at the repo root since the team consolidated them.
# The pinned wheels for CUDA (torch==2.11.0+cu128, torchaudio==2.11.0+cu128)
# and the matching --extra-index-url are stripped before install, since CPU
# torch is already installed above.
COPY --chown=user:user requirements.txt /tmp/requirements.txt
RUN sed -i \
        -e '/^--extra-index-url.*cu128/d' \
        -e '/^torch==.*+cu128/d' \
        -e '/^torchaudio==.*+cu128/d' \
        -e '/^pywinpty/d' \
        -e '/^pywin32/d' \
        /tmp/requirements.txt \
 && pip install --no-cache-dir --user -r /tmp/requirements.txt

# SCAPES module dependencies.
COPY --chown=user:user modules/scapes/requirements.txt /tmp/scapes-requirements.txt
RUN pip install --no-cache-dir --user -r /tmp/scapes-requirements.txt

# ---- App code -------------------------------------------------------------
COPY --chown=user:user app/backend ./app/backend
COPY --chown=user:user modules ./modules

# scapes_runtime.py expects modules/scapes to be a sibling of app/backend,
# resolved from repo_root = parents[3] of the file. The COPY layout above
# satisfies that.

# ---- Runtime assets (model checkpoints + source WAVs) ---------------------
# Pulled from a public HF model repo so the Space repo itself stays small.
# Layout in the assets repo mirrors the in-container target paths:
#   models/Full_150e/checkpoints/*    ->  inference/models/Full_150e/checkpoints/*
#   assets/*.wav                      ->  inference/assets/*.wav
# Override ASSETS_REPO at build time (--build-arg) to point at a fork.
ARG ASSETS_REPO=JordiFabregat1/s103-assets
ARG ASSETS_REVISION=main
ENV ASSETS_REPO=${ASSETS_REPO} \
    ASSETS_REVISION=${ASSETS_REVISION}
RUN python -c "import os; from huggingface_hub import snapshot_download; \
    snapshot_download( \
        repo_id=os.environ['ASSETS_REPO'], \
        repo_type='model', \
        revision=os.environ['ASSETS_REVISION'], \
        local_dir='/home/user/app/app/backend/inference', \
    )" \
 && rm -rf /home/user/app/app/backend/inference/.cache

WORKDIR /home/user/app/app/backend

EXPOSE 7860

# Bind to all interfaces; HF Spaces routes external traffic to 7860.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
