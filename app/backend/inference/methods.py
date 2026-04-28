import os
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import soundfile as sf
import torch
from inference.models import AudioElement, InterpolationElement
from inference.scapes_runtime import (
    CLAPWrapper,
    EncodecProcessor,
    FlowInference,
    load_flow_model,
    load_local_encoder,
    run_interpolation_pipeline,
)


ASSETS_DIR = Path(__file__).resolve().parent / "assets"
MODEL_DIR = Path(__file__).resolve().parent / "models" / "Full_150e"

AUDIO_ASSET_MAP = {
    AudioElement.CAMPFIRE: ASSETS_DIR / "camp_fire.wav",
    AudioElement.KEYBOARD: ASSETS_DIR / "keyboard.wav",
}

FLOW_MODEL_CKPT = Path(os.getenv("SCAPES_FLOW_MODEL_CKPT", MODEL_DIR / "checkpoints" / "best_flow_model.pt"))
FLOW_MODEL_CONFIG = Path(os.getenv("SCAPES_FLOW_MODEL_CONFIG", MODEL_DIR / "checkpoints" / "flow_model_config.json"))
LOCAL_ENCODER_CKPT = Path(os.getenv("SCAPES_LOCAL_ENCODER_CKPT", MODEL_DIR / "checkpoints" / "best_local_encoder.pt"))
LOCAL_ENCODER_CONFIG = Path(os.getenv("SCAPES_LOCAL_ENCODER_CONFIG", MODEL_DIR / "checkpoints" / "local_encoder_config.json"))

ATOMS_FRAMES = int(os.getenv("SCAPES_ATOMS_FRAMES", "48"))
ATOMS_HOP_FRAMES = int(os.getenv("SCAPES_ATOMS_HOP_FRAMES", "15"))
CROSSFADE_FRAMES = int(os.getenv("SCAPES_CROSSFADE_FRAMES", "3"))


def greet() -> str:
    return "Hello from SCAPES Interface!"


def _resolve_audio_path(audio: AudioElement) -> Path:
    audio_path = AUDIO_ASSET_MAP[audio]
    if not audio_path.exists():
        raise FileNotFoundError(
            f"Missing audio asset for '{audio.value}'. Expected file at: {audio_path}"
        )
    return audio_path


def _validate_model_artifacts() -> None:
    missing_paths = [
        path
        for path in [FLOW_MODEL_CKPT, FLOW_MODEL_CONFIG, LOCAL_ENCODER_CKPT, LOCAL_ENCODER_CONFIG]
        if not path.exists()
    ]
    if missing_paths:
        formatted = "\n".join(f"- {path}" for path in missing_paths)
        raise FileNotFoundError(
            "Missing SCAPES model artifacts required for interpolation:\n" + formatted
        )


@lru_cache(maxsize=1)
def get_inference_engine() -> FlowInference:
    _validate_model_artifacts()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = EncodecProcessor(sr=48000, streamable=True, device=device)
    local_encoder = load_local_encoder(
        checkpoint_path=LOCAL_ENCODER_CKPT,
        json_path=LOCAL_ENCODER_CONFIG,
        device=device,
    )
    flow_model = load_flow_model(
        checkpoint_path=FLOW_MODEL_CKPT,
        json_path=FLOW_MODEL_CONFIG,
        device=device,
    )
    clap_model = CLAPWrapper(version="2023", use_cuda=(device == "cuda"))

    return FlowInference(
        model=flow_model,
        local_encoder=local_encoder,
        processor=processor,
        context_model=clap_model,
        segment_length=5,
        context_length=5,
        atoms_frames=ATOMS_FRAMES,
        atoms_hop_frames=ATOMS_HOP_FRAMES,
        crossfade_frames=CROSSFADE_FRAMES,
        device=device,
        verbose=False,
    )


def _waveform_to_wav_bytes(audio_tensor: torch.Tensor, sample_rate: int) -> bytes:
    if audio_tensor.dim() == 3:
        audio_tensor = audio_tensor.squeeze(0)
    if audio_tensor.dim() != 2:
        raise ValueError(f"Expected decoded audio to be 2D, got shape {tuple(audio_tensor.shape)}")

    audio_np = audio_tensor.detach().cpu().float().numpy().T
    buffer = BytesIO()
    sf.write(buffer, audio_np, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def render_interpolation_audio(request: InterpolationElement) -> bytes:
    engine = get_inference_engine()
    audio_path_1 = _resolve_audio_path(request.audio1)
    audio_path_2 = _resolve_audio_path(request.audio2)

    final_audio = run_interpolation_pipeline(
        engine=engine,
        audio_path_1=str(audio_path_1),
        audio_path_2=str(audio_path_2),
        timeline_size=request.timeline_size,
        stay_time=request.stay_time,
        stickyness=request.stickyness,
        plot_stickyness_curve=False,
        play=False,
        save_path=None,
        NFE=request.NFE,
        context_static=request.context_static,
        decode_method="ola_smooth",
        cache=True,
    )
    return _waveform_to_wav_bytes(final_audio, engine.sr)

