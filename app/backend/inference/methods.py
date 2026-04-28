import os
import logging
import time
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

from inference.constants import (
    ASSETS_DIR,
    AUDIO_ASSET_MAP,
    ATOMS_FRAMES,
    ATOMS_HOP_FRAMES,
    CROSSFADE_FRAMES,
    FLOW_MODEL_CKPT,
    FLOW_MODEL_CONFIG,
    LOCAL_ENCODER_CKPT,
    LOCAL_ENCODER_CONFIG,
    MODEL_DIR,
)

logger = logging.getLogger(__name__)


def greet() -> str:
    return "Hello from SCAPES Interface!"


def _resolve_audio_path(audio: AudioElement) -> Path:
    audio_path = AUDIO_ASSET_MAP[audio]
    if not audio_path.exists():
        logger.warning(f"Audio asset not found: {audio_path}")
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
        logger.error(f"Missing model artifacts:\n{formatted}")
        raise FileNotFoundError(
            "Missing SCAPES model artifacts required for interpolation:\n" + formatted
        )


@lru_cache(maxsize=1)
def get_inference_engine() -> FlowInference:
    _validate_model_artifacts()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Initializing inference engine on device: {device}")
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

    logger.info("Running interpolation pipeline...")
    start_time = time.time()
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
    duration = time.time() - start_time
    logger.info(f"Interpolation pipeline finished in {duration:.2f} seconds.")
    return _waveform_to_wav_bytes(final_audio, engine.sr)

