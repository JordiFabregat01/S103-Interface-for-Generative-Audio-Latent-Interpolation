import math
import logging
import os
import threading
import time
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Callable, List, Optional, Tuple

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
from inference.interpolation import EncodedSource, interpolate_clips
from inference.source_cache import (
    encode_and_cache,
    get_or_encode,
    load_encoded_source,
)

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


def _run_interpolation(
    request: InterpolationElement,
    *,
    cancel_event: Optional[threading.Event] = None,
    progress: Optional[Callable[[int, int], None]] = None,
):
    """Drive ``interpolate_clips`` for an :class:`InterpolationElement` payload."""
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
        cancel_event=cancel_event,
        progress=progress,
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
    return result


def _render_interpolation_tensor(
    request: InterpolationElement,
    *,
    cancel_event: Optional[threading.Event] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Tuple[torch.Tensor, int]:
    """Thin wrapper around ``_run_interpolation`` returning ``(audio[C, T], sample_rate)``.

    Needed by ``render_timeline_audio``.
    """
    result = _run_interpolation(request, cancel_event=cancel_event, progress=progress)
    audio = result.audio
    if audio.dim() == 3:
        audio = audio.squeeze(0)
    if audio.dim() != 2:
        raise ValueError(
            f"Expected interpolation audio to be 2D [C, T], got shape {tuple(audio.shape)}"
        )
    return audio, result.sample_rate


def render_interpolation_audio(request: InterpolationElement) -> bytes:
    """Synchronous render path used by the legacy ``/interpolate`` endpoint."""
    result = _run_interpolation(request)
    return _waveform_to_wav_bytes(result.audio, result.sample_rate)


def _get_or_encode_source_by_filename(filename: str) -> EncodedSource:
    """Cache-aware source loader keyed by the WAV file's stem.

    Mirrors :func:`get_or_encode` but accepts any asset filename instead of the
    ``AudioElement`` enum, so timeline clip segments (which only carry the raw
    filename) reuse the same on-disk cache.
    """
    path = resolve_audio_file(filename)
    source_id = path.stem
    try:
        return load_encoded_source(source_id)
    except FileNotFoundError:
        logger.info("source cache miss for %r; encoding now", source_id)
        engine = get_inference_engine()
        return encode_and_cache(engine, path, source_id)


def _generate_clip_tensor(
    filename: str,
    duration_sec: float,
    *,
    cancel_event: Optional[threading.Event] = None,
    nfe: int = 8,
    decode_method: str = "ola_smooth",
) -> Tuple[torch.Tensor, int]:
    """Generate ``duration_sec`` of audio for a clip via the SCAPES model.

    Conditions the generator on the clip's own context embeddings (a sliding
    window starting at ``contexts[0]``, padded with the last context if the
    requested duration exceeds the source). Returns ``([C, T], sample_rate)``.

    This replaces the previous "load WAV from disk" path so every segment in a
    timeline — clip *and* interpolation — goes through the same generative
    pipeline, eliminating timbre discontinuities at the boundaries.
    """
    if duration_sec <= 0:
        raise ValueError(f"duration_sec must be > 0 (got {duration_sec})")

    engine = get_inference_engine()
    source = _get_or_encode_source_by_filename(filename)
    contexts = source.contexts
    if not contexts:
        raise ValueError(f"source {filename!r} has no contexts")

    hop_sec = engine.hop_samples / engine.sr
    timeline_size = max(1, int(round(duration_sec / hop_sec)))

    window = list(contexts[:timeline_size])
    if len(window) < timeline_size:
        missing = timeline_size - len(window)
        logger.warning(
            "clip %s has %d contexts but %d requested; padding with last context",
            filename,
            len(contexts),
            timeline_size,
        )
        window.extend([contexts[-1]] * missing)

    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("cancelled by client")

    contexts_dev = [c.to(engine.device) for c in window]
    timeline = engine.build_base_timeline(
        atoms_129D=[None] * timeline_size,
        context_embeddings=contexts_dev,
        default_TF=False,
        default_AF=0.0,
    )

    start_time = time.time()
    timeline = engine.generate(timeline, NFE=nfe)
    elapsed = time.time() - start_time

    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("cancelled by client")

    audio = engine.decode_timeline(timeline, method=decode_method)
    audio = audio.clamp(-1.0, 1.0).cpu()
    if audio.dim() == 3:
        audio = audio.squeeze(0)
    if audio.dim() != 2:
        raise ValueError(
            f"Expected generated clip audio to be 2D [C, T], got shape {tuple(audio.shape)}"
        )

    logger.info(
        "generated clip %s: timeline_size=%d, duration_sec=%.3f, took %.2fs",
        filename,
        timeline_size,
        timeline_size * hop_sec,
        elapsed,
    )
    return audio, engine.sr


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


def render_timeline_audio(
    request: RenderRequest,
    *,
    cancel_event: Optional[threading.Event] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> bytes:
    """Render a clip / silence / interpolation timeline to a 16-bit PCM WAV.

    Every clip and interpolation segment runs through the SCAPES generator so
    the whole timeline shares a single timbral fingerprint — no seam between
    raw recording and model-generated bridge. Clip segments are conditioned on
    that clip's own context embeddings (no slerp); interpolation segments
    slerp between two clips' contexts as before.

    Interpolation anchors are pinned to the surrounding clips' cut edges so
    every bridge opens at end-of-A and closes at start-of-B. Negative
    ``distance_sec`` = overlap: neighbor clips are trimmed by ``|distance_sec|``
    and ``a_anchor`` shifts back so the dynamic window lands on the cut.

    ``progress(done, total)`` ticks over every generated segment (clips +
    interpolations); silence segments don't move the bar. ``cancel_event``,
    if set between segments, raises ``RuntimeError`` so the job worker can mark
    the job ``cancelled``.
    """
    engine = get_inference_engine()
    sample_rate = engine.sr
    segs = request.segments

    head_trim_samples = [0] * len(segs)
    tail_trim_samples = [0] * len(segs)
    anchor_overrides: dict[int, Tuple[float, float]] = {}

    for idx, segment in enumerate(segs):
        if segment.type != "interpolation":
            continue

        a_anchor = segment.a_anchor_sec
        b_anchor = segment.b_anchor_sec

        prev_seg = segs[idx - 1] if idx > 0 else None
        next_seg = segs[idx + 1] if idx + 1 < len(segs) else None

        if segment.distance_sec < 0:
            overlap_sec = -segment.distance_sec
            overlap_samples = int(round(overlap_sec * sample_rate))
            if prev_seg is not None and prev_seg.type == "clip":
                tail_trim_samples[idx - 1] = max(
                    tail_trim_samples[idx - 1], overlap_samples
                )
                a_anchor = max(0.0, prev_seg.duration - overlap_sec)
            if next_seg is not None and next_seg.type == "clip":
                head_trim_samples[idx + 1] = max(
                    head_trim_samples[idx + 1], overlap_samples
                )
                b_anchor = 0.0
        else:
            if prev_seg is not None and prev_seg.type == "clip":
                a_anchor = prev_seg.duration
            if next_seg is not None and next_seg.type == "clip":
                b_anchor = 0.0

        anchor_overrides[idx] = (a_anchor, b_anchor)

    generable_total = sum(1 for s in segs if s.type in ("clip", "interpolation"))
    generable_done = 0
    if progress is not None:
        progress(generable_done, generable_total)

    tensors: List[torch.Tensor] = []
    for index, segment in enumerate(segs):
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("cancelled by client")

        if segment.type == "clip":
            tensor, seg_sr = _generate_clip_tensor(
                segment.filename,
                segment.duration,
                cancel_event=cancel_event,
            )
            if seg_sr != sample_rate:
                raise ValueError(
                    f"clip sample rate {seg_sr} != timeline {sample_rate}"
                )
            total = tensor.shape[-1]
            start = min(head_trim_samples[index], total)
            end = max(start, total - tail_trim_samples[index])
            if end <= start:
                logger.warning(
                    "clip %s at index %d fully consumed by adjacent overlap; dropping",
                    segment.filename,
                    index,
                )
                continue
            if start > 0 or end < total:
                tensor = tensor[..., start:end]
            generable_done += 1
            if progress is not None:
                progress(generable_done, generable_total)
        elif segment.type == "silence":
            tensor = _make_silence(segment.duration, sample_rate=sample_rate)
        elif segment.type == "interpolation":
            element = segment.to_element()
            if index in anchor_overrides:
                a_anchor, b_anchor = anchor_overrides[index]
                element = element.model_copy(
                    update={"a_anchor_sec": a_anchor, "b_anchor_sec": b_anchor}
                )
            tensor, seg_sr = _render_interpolation_tensor(
                element, cancel_event=cancel_event
            )
            if seg_sr != sample_rate:
                raise ValueError(
                    f"interpolation sample rate {seg_sr} != timeline {sample_rate}"
                )
            generable_done += 1
            if progress is not None:
                progress(generable_done, generable_total)
        else:  # pragma: no cover - guarded by the discriminated union
            raise ValueError(f"unknown segment type at index {index}")

        if tensor.shape[-1] > 0:
            tensors.append(tensor)

    if not tensors:
        raise ValueError("timeline produced no audio")

    channels = max(seg.shape[0] for seg in tensors)
    stitched = torch.cat(
        [_match_channels(seg, channels) for seg in tensors], dim=-1
    )
    logger.info(
        "rendered timeline: %d segments, %d channels, %.3fs",
        len(segs),
        channels,
        stitched.shape[-1] / sample_rate,
    )
    return _waveform_to_wav_bytes(stitched, sample_rate)


def render_interpolation_to_file(
    request: InterpolationElement,
    output_path: Path,
    *,
    cancel_event: Optional[threading.Event] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> int:
    """Render an interpolation and atomically persist the WAV to ``output_path``.

    Returns the size in bytes of the written file. Used by the async ``/render``
    job runner; supports cooperative cancellation through ``cancel_event`` and
    progress reporting via ``progress(done, total)``.
    """
    result = _run_interpolation(
        request, cancel_event=cancel_event, progress=progress
    )
    audio_bytes = _waveform_to_wav_bytes(result.audio, result.sample_rate)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_bytes(audio_bytes)
    os.replace(tmp_path, output_path)
    return len(audio_bytes)


def render_timeline_to_file(
    request: RenderRequest,
    output_path: Path,
    *,
    cancel_event: Optional[threading.Event] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> int:
    """Render a timeline and atomically persist the WAV to ``output_path``.

    Runner-compatible companion to :func:`render_timeline_audio`: forwards
    the job ``cancel_event`` / ``progress`` callback so the bar advances
    per interpolation segment and cancel is honored between segments.
    """
    audio_bytes = render_timeline_audio(
        request, cancel_event=cancel_event, progress=progress
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_bytes(audio_bytes)
    os.replace(tmp_path, output_path)
    return len(audio_bytes)

