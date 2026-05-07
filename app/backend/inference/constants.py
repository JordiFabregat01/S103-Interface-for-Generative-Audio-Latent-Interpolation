import os
from pathlib import Path

from inference.models import AudioElement

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
MODEL_DIR = Path(__file__).resolve().parent / "models" / "Full_150e"

CACHE_DIR = ASSETS_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

AUDIO_ASSET_MAP = {
    AudioElement.CAMPFIRE: ASSETS_DIR / "camp_fire.wav",
    AudioElement.KEYBOARD: ASSETS_DIR / "keyboard.wav",
}
AUDIO_CACHE_MAP = {
    AudioElement.CAMPFIRE: "camp_fire",
    AudioElement.KEYBOARD: "keyboard",
}

FLOW_MODEL_CKPT = Path(os.getenv("SCAPES_FLOW_MODEL_CKPT", MODEL_DIR / "checkpoints" / "best_flow_model.pt"))
FLOW_MODEL_CONFIG = Path(os.getenv("SCAPES_FLOW_MODEL_CONFIG", MODEL_DIR / "checkpoints" / "flow_model_config.json"))
LOCAL_ENCODER_CKPT = Path(os.getenv("SCAPES_LOCAL_ENCODER_CKPT", MODEL_DIR / "checkpoints" / "best_local_encoder.pt"))
LOCAL_ENCODER_CONFIG = Path(os.getenv("SCAPES_LOCAL_ENCODER_CONFIG", MODEL_DIR / "checkpoints" / "local_encoder_config.json"))

ATOMS_FRAMES = int(os.getenv("SCAPES_ATOMS_FRAMES", "48"))
ATOMS_HOP_FRAMES = int(os.getenv("SCAPES_ATOMS_HOP_FRAMES", "15"))
CROSSFADE_FRAMES = int(os.getenv("SCAPES_CROSSFADE_FRAMES", "3"))

