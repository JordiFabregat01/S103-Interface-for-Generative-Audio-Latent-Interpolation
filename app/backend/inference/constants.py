import os
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
MODEL_DIR = Path(__file__).resolve().parent / "models" / "Full_150e"

CACHE_DIR = ASSETS_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Library sounds live in per-kind subfolders so the "short" cherrypick and
# "long" original variants (which can share a stem, e.g. BreakingWater) don't
# collide. A sound is identified by (name, kind); the canonical cache key joins
# them so the two variants get distinct .source.pt / embedding entries.
SOUND_KINDS = ("short", "long")


def audio_asset_path(name: str, kind: str) -> Path:
    """Absolute path to a library wav given its stem and kind."""
    return ASSETS_DIR / kind / f"{name}.wav"


def source_cache_key(name: str, kind: str) -> str:
    """Canonical, collision-free source id, e.g. ``short__BreakingWater``."""
    return f"{kind}__{name}"

FLOW_MODEL_CKPT = Path(os.getenv("SCAPES_FLOW_MODEL_CKPT", MODEL_DIR / "checkpoints" / "best_flow_model.pt"))
FLOW_MODEL_CONFIG = Path(os.getenv("SCAPES_FLOW_MODEL_CONFIG", MODEL_DIR / "checkpoints" / "flow_model_config.json"))
LOCAL_ENCODER_CKPT = Path(os.getenv("SCAPES_LOCAL_ENCODER_CKPT", MODEL_DIR / "checkpoints" / "best_local_encoder.pt"))
LOCAL_ENCODER_CONFIG = Path(os.getenv("SCAPES_LOCAL_ENCODER_CONFIG", MODEL_DIR / "checkpoints" / "local_encoder_config.json"))

ATOMS_FRAMES = int(os.getenv("SCAPES_ATOMS_FRAMES", "48"))
ATOMS_HOP_FRAMES = int(os.getenv("SCAPES_ATOMS_HOP_FRAMES", "15"))
CROSSFADE_FRAMES = int(os.getenv("SCAPES_CROSSFADE_FRAMES", "3"))

