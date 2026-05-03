import math
import os
import logging
import time
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple

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

def time_to_sample_indices(
    start_sec: Optional[float],
    end_sec: Optional[float],
    *,
    sample_rate: int,
    num_samples: int,
) -> Tuple[int, int]:
    
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")
    if start_sec is not None and start_sec < 0:
        raise ValueError("start_sec must be non-negative")
    if end_sec is not None and end_sec < 0:
        raise ValueError("end_sec must be non-negative")

    start_sample = (
        0 if start_sec is None else max(0, min(num_samples, int(round(float(start_sec) * sample_rate))))
    )
    end_sample = (
        num_samples
        if end_sec is None
        else max(0, min(num_samples, int(round(float(end_sec) * sample_rate))))
    )

    if start_sample >= end_sample:
        raise ValueError(
            f"empty or inverted sample range: start_sample={start_sample}, end_sample={end_sample}"
        )
    return start_sample, end_sample


def time_to_atom_indices(
    start_sec: Optional[float],
    end_sec: Optional[float],
    *,
    engine: FlowInference,
    num_atoms: int,
) -> Tuple[int, int]:
    
    if num_atoms <= 0:
        raise ValueError("num_atoms must be positive")
    if start_sec is not None and start_sec < 0:
        raise ValueError("start_sec must be non-negative")
    if end_sec is not None and end_sec < 0:
        raise ValueError("end_sec must be non-negative")

    sr = engine.sr
    hop = engine.hop_samples
    seg = engine.segment_samples
    max_end_sample = (num_atoms - 1) * hop + seg

    start_sample = 0 if start_sec is None else max(0, int(round(float(start_sec) * sr)))
    if end_sec is None:
        end_sample = max_end_sample
    else:
        end_sample = min(max_end_sample, int(round(float(end_sec) * sr)))

    if start_sample >= end_sample:
        raise ValueError(
            f"empty or inverted range after mapping to samples: "
            f"start_sample={start_sample}, end_sample={end_sample}"
        )

    lo = max(0, (start_sample - seg + hop) // hop)
    hi = min(num_atoms, math.ceil(end_sample / hop))
    if hi <= lo:
        raise ValueError(
            "time window maps to fewer than one atom hop; widen the selection."
        )
    return lo, hi

    
def trim_atoms_contexts(
    atoms: list,
    contexts: list,
    start_sec: Optional[float],
    end_sec: Optional[float],
    *,
    engine: FlowInference,
) -> Tuple[list, list]:
    """
    Trim atoms and contexts based on start and end times (in seconds).

    Returns:
        tuple[list, list]: Sliced atoms and contexts between the corresponding atom indices.
    """
    if len(atoms) != len(contexts):
        raise ValueError(
            f"atoms length ({len(atoms)}) must match contexts length ({len(contexts)})"
        )

    # Use time_to_atom_indices to determine the atom index range
    lo, hi = time_to_atom_indices(
        start_sec, end_sec,
        engine=engine,
        num_atoms=len(atoms),
    )
    if lo < 0 or hi < 0:
        raise ValueError("Computed atom indices must be non-negative")
    if lo > len(atoms) or hi > len(atoms):
        raise ValueError(
            f"indices out of range for length {len(atoms)}: start_index={lo}, end_index={hi}"
        )
    if lo >= hi:
        raise ValueError(
            f"empty slice: start_index ({lo}) must be < end_index ({hi})"
        )
    return atoms[lo:hi], contexts[lo:hi]


def trim_waveform(
    audio_tensor: torch.Tensor,
    start_sample: int,
    end_sample: Optional[int] = None,
) -> torch.Tensor:
    """
    Slice the time dimension like ``audio_tensor[..., start_sample:end_sample]``.

    Accepts shapes ending with time ``T``: ``[1, C, T]`` (SCAPES) or ``[C, T]``.
    ``end_sample`` defaults to ``T`` (through last sample).
    """
    if audio_tensor.dim() not in (2, 3):
        raise ValueError(
            f"expected waveform [1, C, T] or [C, T], got shape {tuple(audio_tensor.shape)}"
        )
    n = audio_tensor.shape[-1]
    if end_sample is None:
        end_sample = n
    if start_sample < 0 or end_sample < 0:
        raise ValueError("start_sample and end_sample must be non-negative")
    if start_sample > n or end_sample > n:
        raise ValueError(
            f"sample indices out of range for length {n}: "
            f"start_sample={start_sample}, end_sample={end_sample}"
        )
    if start_sample >= end_sample:
        raise ValueError(
            f"empty slice: start_sample ({start_sample}) must be < end_sample ({end_sample})"
        )
    return audio_tensor[..., start_sample:end_sample]




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

