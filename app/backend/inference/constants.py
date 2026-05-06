import os
from pathlib import Path
from dotenv import load_dotenv

from inference.models import AudioElement

load_dotenv()

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

CACHE_DIR = ASSETS_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

def _list_available_model_dirs() -> list[Path]:
    """List all subdirectories in WEIGHTS_DIR, ignoring files like .gitkeep."""
    if not WEIGHTS_DIR.exists():
        return []
    return [p for p in WEIGHTS_DIR.iterdir() if p.is_dir()]

def get_model_dir() -> Path:
    """Get the active model directory based on SCAPES_MODEL_NAME or auto-selection."""
    model_dirs = _list_available_model_dirs()
    if not model_dirs:
        raise FileNotFoundError(
            f"No model directories found in {WEIGHTS_DIR}. "
            "Please ensure at least one model folder exists under 'weights/'."
        )
    
    model_name = os.getenv("SCAPES_MODEL_NAME")
    if model_name:
        selected_dir = WEIGHTS_DIR / model_name
        if selected_dir in model_dirs:
            return selected_dir
        else:
            available_names = [d.name for d in model_dirs]
            raise ValueError(
                f"SCAPES_MODEL_NAME='{model_name}' does not match any available model directory. "
                f"Available models: {', '.join(available_names)}"
            )
    
    # No override: if exactly one model, use it; otherwise, require selection
    if len(model_dirs) == 1:
        return model_dirs[0]
    else:
        available_names = [d.name for d in model_dirs]
        raise ValueError(
            f"Multiple model directories found: {', '.join(available_names)}. "
            "Please set SCAPES_MODEL_NAME in .env to select one."
        )
def _get_audio_asset_path(audio: AudioElement) -> Path:
    return ASSETS_DIR / f"{audio.value}.wav"

def _get_audio_cache_key(audio: AudioElement) -> str:
    return audio.value

FLOW_MODEL_CKPT = Path(os.getenv("SCAPES_FLOW_MODEL_CKPT", get_model_dir() / "checkpoints" / "best_flow_model.pt"))
FLOW_MODEL_CONFIG = Path(os.getenv("SCAPES_FLOW_MODEL_CONFIG", get_model_dir() / "checkpoints" / "flow_model_config.json"))
LOCAL_ENCODER_CKPT = Path(os.getenv("SCAPES_LOCAL_ENCODER_CKPT", get_model_dir() / "checkpoints" / "best_local_encoder.pt"))
LOCAL_ENCODER_CONFIG = Path(os.getenv("SCAPES_LOCAL_ENCODER_CONFIG", get_model_dir() / "checkpoints" / "local_encoder_config.json"))

ATOMS_FRAMES = int(os.getenv("SCAPES_ATOMS_FRAMES", "21"))
ATOMS_HOP_FRAMES = int(os.getenv("SCAPES_ATOMS_HOP_FRAMES", "15"))
CROSSFADE_FRAMES = int(os.getenv("SCAPES_CROSSFADE_FRAMES", "3"))

