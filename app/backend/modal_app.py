import modal

# ── Image ─────────────────────────────────────────────────────────────────────
# Code is baked into the image so scapes_runtime.py's parents[3] resolves to /repo.
# Layout inside container:
#   /repo/app/backend/inference/scapes_runtime.py  (parents[3] == /repo)
#   /repo/modules/scapes/
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch", "torchaudio",
        extra_index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install(
        "torchdiffeq", "librosa", "numpy", "matplotlib", "soundfile",
        "transformers", "tqdm", "msclap", "fastapi", "uvicorn",
        "pydantic", "scikit-learn", "ipython", "huggingface_hub",
    )
    .add_local_dir("app/backend", remote_path="/repo/app/backend")
    .add_local_dir("modules", remote_path="/repo/modules")
)

# ── Volume (model checkpoints + source WAVs) ──────────────────────────────────
# Downloaded once via `modal run modal_app.py::download_assets`, then reused.
volume = modal.Volume.from_name("s103-assets", create_if_missing=True)
VOLUME_PATH = "/assets"

app = modal.App("s103-backend", image=image)


@app.function(
    volumes={VOLUME_PATH: volume},
    # Uncomment for GPU inference:
    # gpu="t4",
    timeout=300,
    scaledown_window=60,
    memory=4096,
    secrets=[modal.Secret.from_name("s103-secrets")],
)
@modal.asgi_app()
def fastapi_app():
    import os
    import sys

    sys.path.insert(0, "/repo/app/backend")

    os.environ["SCAPES_ASSETS_DIR"] = f"{VOLUME_PATH}/assets"
    os.environ["SCAPES_MODEL_DIR"] = f"{VOLUME_PATH}/models/Full_150e"

    from main import app as fastapi_app
    return fastapi_app


@app.function(volumes={VOLUME_PATH: volume})
def download_assets(
    repo_id: str = "JordiFabregat1/s103-assets",
    revision: str = "main",
):
    """Pull model checkpoints and WAV files into the Modal volume.

    Run once before deploying:
        modal run app/backend/modal_app.py::download_assets
    """
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        local_dir=VOLUME_PATH,
    )
    volume.commit()
    print(f"Assets downloaded from {repo_id}@{revision} -> {VOLUME_PATH}")
