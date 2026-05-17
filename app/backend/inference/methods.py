import math
import logging
import time
from functools import lru_cache
from io import BytesIO
from typing import List, Optional, Tuple

import librosa
import soundfile as sf
import torch
from inference.models import InterpolationElement, RenderRequest
from inference.embeddings import resolve_audio_file
from inference.scapes_runtime import (
    CLAPWrapper,
    EncodecProcessor,
    FlowInference,
    load_flow_model,
    load_local_encoder,
)
from inference.interpolation import interpolate_clips
from inference.source_cache import get_or_encode

from inference.constants import (
    ATOMS_FRAMES,
    ATOMS_HOP_FRAMES,
    CROSSFADE_FRAMES,
    FLOW_MODEL_CKPT,
    FLOW_MODEL_CONFIG,
    LOCAL_ENCODER_CKPT,
    LOCAL_ENCODER_CONFIG,
)

logger = logging.getLogger(__name__)


def greet() -> str:
    return "Hello from SCAPES Interface!"


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
        raise ValueError(
            f"Expected decoded audio to be 2D, got shape {tuple(audio_tensor.shape)}"
        )

    peak = float(audio_tensor.detach().abs().max().item()) if audio_tensor.numel() else 0.0
    if peak > 1.0:
        logger.warning(
            "audio peak %.3f exceeds 1.0; clamping to [-1, 1] before WAV write",
            peak,
        )
    audio_tensor = audio_tensor.clamp(-1.0, 1.0)

    audio_np = audio_tensor.detach().cpu().float().numpy().T
    buffer = BytesIO()
    sf.write(buffer, audio_np, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _render_interpolation_tensor(request: InterpolationElement) -> Tuple[torch.Tensor, int]:
    """Run the interpolation pipeline and return ``(audio[C, T], sample_rate)``."""
    engine = get_inference_engine()

    src_a = get_or_encode(engine, request.audio1)
    src_b = get_or_encode(engine, request.audio2)

    start_time = time.time()
    result = interpolate_clips(
        engine,
        src_a,
        src_b,
        request.distance_sec,
        adjacent_duration_sec=request.duration_sec,
        a_anchor_sec=request.a_anchor_sec,
        b_anchor_sec=request.b_anchor_sec,
        stay_time_sec=request.stay_time_sec,
        stickyness=request.stickyness,
        nfe=request.nfe,
        decode_method=request.decode_method,
        context_mode_override=request.context_mode,
        # cancel_event / progress not wired to the synchronous endpoints yet;
        # left as None until a streaming endpoint exists.
    )
    elapsed = time.time() - start_time
    logger.info(
        "interpolation finished in %.2fs (timeline_size=%d, context_mode=%s, "
        "duration_sec=%.3f)",
        elapsed,
        result.timeline_size,
        result.context_mode,
        result.duration_sec,
    )

    audio = result.audio
    if audio.dim() == 3:
        audio = audio.squeeze(0)
    if audio.dim() != 2:
        raise ValueError(
            f"Expected interpolation audio to be 2D [C, T], got shape {tuple(audio.shape)}"
        )
    return audio, result.sample_rate


def render_interpolation_audio(request: InterpolationElement) -> bytes:
    audio, sample_rate = _render_interpolation_tensor(request)
    return _waveform_to_wav_bytes(audio, sample_rate)


def _load_clip_tensor(
    filename: str, duration_sec: float, *, sample_rate: int
) -> torch.Tensor:
    """Load a clip WAV as ``[C, T]`` at ``sample_rate``, trimmed/padded to ``duration_sec``."""
    path = resolve_audio_file(filename)
    data, file_sr = sf.read(str(path), always_2d=True, dtype="float32")  # [T, C]
    audio = torch.from_numpy(data.T).contiguous()  # [C, T]

    if file_sr != sample_rate:
        resampled = librosa.resample(
            audio.numpy(), orig_sr=file_sr, target_sr=sample_rate, axis=-1
        )
        audio = torch.from_numpy(resampled).contiguous()

    target_len = int(round(duration_sec * sample_rate))
    cur_len = audio.shape[-1]
    if cur_len >= target_len:
        return audio[..., :target_len]

    logger.info(
        "clip %s shorter than requested %.3fs (%d < %d samples); zero-padding",
        filename,
        duration_sec,
        cur_len,
        target_len,
    )
    pad = torch.zeros(audio.shape[0], target_len - cur_len, dtype=audio.dtype)
    return torch.cat([audio, pad], dim=-1)


def _make_silence(duration_sec: float, *, sample_rate: int) -> torch.Tensor:
    """Mono silence ``[1, T]``; channel count is reconciled during stitching."""
    n = int(round(duration_sec * sample_rate))
    return torch.zeros(1, max(0, n), dtype=torch.float32)


def _match_channels(audio: torch.Tensor, channels: int) -> torch.Tensor:
    """Coerce ``[C, T]`` to exactly ``channels`` rows (upmix mono, downmix to mono)."""
    c = audio.shape[0]
    if c == channels:
        return audio
    if c == 1:
        return audio.expand(channels, -1).contiguous()
    if channels == 1:
        return audio.mean(dim=0, keepdim=True)
    # Uncommon (e.g. 6ch -> 2ch): downmix to mono, then fan out.
    return audio.mean(dim=0, keepdim=True).expand(channels, -1).contiguous()


def render_timeline_audio(request: RenderRequest) -> bytes:
    """Render a full timeline (clips + silence + interpolation) to a WAV byte string."""
    engine = get_inference_engine()
    sample_rate = engine.sr

    segments: List[torch.Tensor] = []
    for index, segment in enumerate(request.segments):
        if segment.type == "clip":
            tensor = _load_clip_tensor(
                segment.filename, segment.duration, sample_rate=sample_rate
            )
        elif segment.type == "silence":
            tensor = _make_silence(segment.duration, sample_rate=sample_rate)
        elif segment.type == "interpolation":
            tensor, seg_sr = _render_interpolation_tensor(segment.to_element())
            if seg_sr != sample_rate:
                raise ValueError(
                    f"interpolation sample rate {seg_sr} != timeline {sample_rate}"
                )
        else:  # pragma: no cover - guarded by the discriminated union
            raise ValueError(f"unknown segment type at index {index}")

        if tensor.shape[-1] > 0:
            segments.append(tensor)

    if not segments:
        raise ValueError("timeline produced no audio")

    channels = max(seg.shape[0] for seg in segments)
    stitched = torch.cat(
        [_match_channels(seg, channels) for seg in segments], dim=-1
    )
    logger.info(
        "rendered timeline: %d segments, %d channels, %.3fs",
        len(request.segments),
        channels,
        stitched.shape[-1] / sample_rate,
    )
    return _waveform_to_wav_bytes(stitched, sample_rate)

