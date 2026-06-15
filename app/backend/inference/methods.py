import math
import logging
import os
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import soundfile as sf
import torch
from inference.models import InterpolationElement, InterpolationSegment, RenderRequest
from inference.embeddings import resolve_audio_file
from inference.scapes_runtime import (
    CLAPWrapper,
    EncodecProcessor,
    FlowInference,
    load_flow_model,
    load_local_encoder,
)
from inference.interpolation import (
    EncodedSource,
    build_interpolation_context_schedule,
    request_from_clip_geometry,
)
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
    source_cache_key,
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


def render_interpolation_audio(request: InterpolationElement) -> bytes:
    """Synchronous render of a single interpolation through the unified path."""
    timeline_request = RenderRequest(
        segments=[InterpolationSegment.from_element(request)]
    )
    return render_timeline_audio(timeline_request)


def _get_or_encode_source_by_filename(filename: str, kind: str) -> EncodedSource:
    """Cache-aware source loader for a timeline clip's ``(filename, kind)``.

    Resolves the wav under ``assets/{kind}/`` and keys the cache by
    ``f"{kind}__{stem}"`` so it shares the exact cache entry the interpolation
    path (``get_or_encode``) would build for the same sound variant.
    """
    path = resolve_audio_file(filename, kind)
    source_id = source_cache_key(path.stem, kind)
    try:
        return load_encoded_source(source_id)
    except FileNotFoundError:
        logger.info("source cache miss for %r; encoding now", source_id)
        engine = get_inference_engine()
        return encode_and_cache(engine, path, source_id)


def build_clip_context_schedule(
    source: EncodedSource,
    duration_sec: float,
    engine: FlowInference,
) -> List[torch.Tensor]:
    """Return one context tensor per atom slot for ``duration_sec`` of clip audio.

    Slices ``source.contexts`` from the front, padding with the final context
    if the request exceeds the source. Tensors are moved to ``engine.device``
    so the unified schedule can be handed straight to ``build_base_timeline``.
    """
    if duration_sec <= 0:
        raise ValueError(f"duration_sec must be > 0 (got {duration_sec})")
    contexts = source.contexts
    if not contexts:
        raise ValueError(f"source {source.source_id!r} has no contexts")

    hop_sec = engine.hop_samples / engine.sr
    timeline_size = max(1, int(round(duration_sec / hop_sec)))

    window = list(contexts[:timeline_size])
    if len(window) < timeline_size:
        missing = timeline_size - len(window)
        logger.warning(
            "clip %s has %d contexts but %d requested; padding with last context",
            source.source_id,
            len(contexts),
            timeline_size,
        )
        window.extend([contexts[-1]] * missing)

    return [c.to(engine.device) for c in window]


def _make_silence_audio(duration_sec: float, *, sample_rate: int) -> torch.Tensor:
    """Mono silence ``[1, T]`` for silence-only timelines that bypass SCAPES."""
    n = int(round(duration_sec * sample_rate))
    return torch.zeros(1, max(0, n), dtype=torch.float32)


@torch.no_grad()
def run_generate_with_progress(
    engine: FlowInference,
    timeline: List[dict],
    *,
    nfe: int = 8,
    cancel_event: Optional[threading.Event] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> List[dict]:
    """Per-atom replication of :meth:`FlowInference.generate` with hooks.

    Mirrors the submodule loop verbatim (FlowInference.py:461-507) and adds
    ``progress(done, total)`` after each atom plus a ``cancel_event`` check
    at the top of each step. The submodule itself is not modified.
    """
    if not timeline:
        raise ValueError("Timeline is empty!")

    engine.model.eval()
    engine.local_encoder.eval()

    M = engine.segment_length
    total_steps = len(timeline)
    dummy_atom = torch.zeros(1, 129, engine.atoms_frames, device=engine.device)

    if progress is not None:
        progress(0, total_steps)

    for t in range(total_steps):
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("cancelled by client")

        past_atoms = []
        for i in range(t - M, t):
            if i < 0:
                past_atoms.append(dummy_atom)
            else:
                step_dict = timeline[i]
                if step_dict["TF"]:
                    past_atoms.append(step_dict["atom_given"].to(engine.device))
                else:
                    past_atoms.append(step_dict["atom_generated"].to(engine.device))

        past_buffer = torch.cat(past_atoms, dim=0).unsqueeze(0)
        encoded_past = engine.local_encoder(past_buffer)

        num_nulls = max(0, M - t)
        if num_nulls > 0:
            encoded_past[:, :num_nulls, :, :] = engine.model.null_past_embed

        context = timeline[t]["context_embedding"].to(engine.device)
        if context.dim() == 1:
            context = context.unsqueeze(0)

        x0 = torch.randn(1, engine.atoms_frames, 129, device=engine.device)
        pred = engine.model.generate(x0, encoded_past, context, max_nfe=nfe)
        timeline[t]["atom_generated"] = pred.transpose(1, 2)

        if progress is not None:
            progress(t + 1, total_steps)

    return timeline


@dataclass
class _SilenceRange:
    """Sample range to zero-fill after decoding."""
    start_sample: int
    end_sample: int


def _neighbor_clip_context(
    segs: List, idx: int, step: int, engine: FlowInference
) -> Optional[torch.Tensor]:
    """Find the nearest clip in ``step`` direction and return one of its contexts.

    ``step = -1`` returns the *last* context of the preceding clip (the natural
    "where the model just left off" vector); ``step = +1`` returns the *first*
    context of the following clip. Falls back to interpolation endpoints if no
    raw clip neighbor exists.
    """
    i = idx + step
    while 0 <= i < len(segs):
        seg = segs[i]
        if seg.type == "clip":
            src = _get_or_encode_source_by_filename(seg.filename, seg.kind)
            ctx = src.contexts[-1] if step < 0 else src.contexts[0]
            return ctx.to(engine.device)
        if seg.type == "interpolation":
            if step < 0:
                element_audio, element_kind = seg.audio2, seg.audio2_kind
            else:
                element_audio, element_kind = seg.audio1, seg.audio1_kind
            src = get_or_encode(engine, element_audio, element_kind)
            ctx = src.contexts[-1] if step < 0 else src.contexts[0]
            return ctx.to(engine.device)
        i += step
    return None


def _interpolation_anchor_override(
    segs: List, idx: int, hop_sec: float
) -> Tuple[Tuple[float, float], int, int]:
    """Compute (a_anchor, b_anchor) and neighbor clip atom-trim for ``segs[idx]``.

    Returns ``((a_anchor, b_anchor), prev_tail_trim_atoms, next_head_trim_atoms)``.
    Mirrors the per-segment overlap math but in atom units, so the unified
    schedule can pre-trim clip atom counts before generation.
    """
    seg = segs[idx]
    a_anchor = seg.a_anchor_sec
    b_anchor = seg.b_anchor_sec
    prev_seg = segs[idx - 1] if idx > 0 else None
    next_seg = segs[idx + 1] if idx + 1 < len(segs) else None

    prev_tail_trim = 0
    next_head_trim = 0

    if seg.distance_sec < 0:
        overlap_sec = -seg.distance_sec
        overlap_atoms = max(1, int(round(overlap_sec / hop_sec)))
        if prev_seg is not None and prev_seg.type == "clip":
            prev_tail_trim = overlap_atoms
            a_anchor = max(0.0, prev_seg.duration - overlap_sec)
        if next_seg is not None and next_seg.type == "clip":
            next_head_trim = overlap_atoms
            b_anchor = 0.0
    else:
        if prev_seg is not None and prev_seg.type == "clip":
            a_anchor = prev_seg.duration
        if next_seg is not None and next_seg.type == "clip":
            b_anchor = 0.0

    return (a_anchor, b_anchor), prev_tail_trim, next_head_trim


def build_unified_timeline_schedule(
    request: RenderRequest, engine: FlowInference
) -> Tuple[List[torch.Tensor], List[_SilenceRange]]:
    """Flatten the timeline into one per-atom context schedule for SCAPES.

    Returns ``(schedule, silence_ranges)``: ``schedule`` is a list of context
    tensors on ``engine.device`` (one per atom slot, end to end) ready to feed
    into ``engine.build_base_timeline``; ``silence_ranges`` is a list of
    ``(start_sample, end_sample)`` ranges in the decoded output that should be
    zero-filled post-decode.

    Clip atom counts incorporate any overlap-trim demanded by adjacent
    interpolation segments (negative ``distance_sec``), so the schedule
    contains exactly the atoms the unified generation should produce — no
    post-generation sample trimming required.
    """
    segs = request.segments
    hop_sec = engine.hop_samples / engine.sr

    head_trim_atoms = [0] * len(segs)
    tail_trim_atoms = [0] * len(segs)
    anchor_overrides: dict = {}
    for idx, seg in enumerate(segs):
        if seg.type != "interpolation":
            continue
        (a_anchor, b_anchor), prev_trim, next_trim = _interpolation_anchor_override(
            segs, idx, hop_sec
        )
        anchor_overrides[idx] = (a_anchor, b_anchor)
        if prev_trim > 0:
            tail_trim_atoms[idx - 1] = max(tail_trim_atoms[idx - 1], prev_trim)
        if next_trim > 0:
            head_trim_atoms[idx + 1] = max(head_trim_atoms[idx + 1], next_trim)

    schedule: List[torch.Tensor] = []
    silence_ranges: List[_SilenceRange] = []

    for idx, seg in enumerate(segs):
        atom_offset = len(schedule)

        if seg.type == "clip":
            source = _get_or_encode_source_by_filename(seg.filename, seg.kind)
            ctx = build_clip_context_schedule(source, seg.duration, engine)
            start = min(head_trim_atoms[idx], len(ctx))
            end = max(start, len(ctx) - tail_trim_atoms[idx])
            if end <= start:
                logger.warning(
                    "clip %s at index %d fully consumed by adjacent overlap; dropping",
                    seg.filename,
                    idx,
                )
                continue
            schedule.extend(ctx[start:end])

        elif seg.type == "interpolation":
            src_a = get_or_encode(engine, seg.audio1, seg.audio1_kind)
            src_b = get_or_encode(engine, seg.audio2, seg.audio2_kind)
            a_anchor, b_anchor = anchor_overrides.get(
                idx, (seg.a_anchor_sec, seg.b_anchor_sec)
            )
            ir = request_from_clip_geometry(
                src_a,
                src_b,
                seg.distance_sec,
                adjacent_duration_sec=seg.duration_sec,
                a_anchor_sec=a_anchor,
                b_anchor_sec=b_anchor,
                stay_time_sec=seg.stay_time_sec,
                stickyness=seg.stickyness,
                nfe=seg.nfe,
                decode_method=seg.decode_method,
                context_mode_override=seg.context_mode,
            )
            ctx, _mode = build_interpolation_context_schedule(engine, ir)
            schedule.extend(ctx)

        elif seg.type == "silence":
            atom_count = max(1, int(round(seg.duration / hop_sec)))
            prev_ctx = _neighbor_clip_context(segs, idx, -1, engine)
            next_ctx = _neighbor_clip_context(segs, idx, +1, engine)

            # Silence is always muted (zero-filled post-decode). The atoms
            # underneath are still generated to keep the autoregressive past
            # buffer warm; the context we feed them controls what timbre flows
            # into the *next* clip. Splitting a between-clips gap into a first
            # half of the preceding sound and a second half of the following
            # sound means the next clip's AR history is already its own timbre,
            # so no remnant of the previous sound bleeds in when its volume
            # comes back up (generation has variance, so an audible tail would
            # otherwise leak across the mute boundary).
            if prev_ctx is not None and next_ctx is not None:
                half = atom_count // 2
                schedule.extend([prev_ctx] * half)
                schedule.extend([next_ctx] * (atom_count - half))
            elif prev_ctx is not None:
                # Trailing silence: nothing follows, so stay on the preceding sound.
                schedule.extend([prev_ctx] * atom_count)
            elif next_ctx is not None:
                # Leading silence: warm the buffer with the following sound.
                schedule.extend([next_ctx] * atom_count)
            else:
                raise ValueError(
                    "silence segment has no neighboring clip or interpolation "
                    "to inherit context from; silence-only timelines bypass SCAPES"
                )

            start_sample = atom_offset * engine.hop_samples
            end_sample = start_sample + int(round(seg.duration * engine.sr))
            silence_ranges.append(_SilenceRange(start_sample, end_sample))

        else:  # pragma: no cover - guarded by the discriminated union
            raise ValueError(f"unknown segment type at index {idx}: {seg.type}")

    return schedule, silence_ranges


def _resolve_render_params(request: RenderRequest) -> Tuple[int, str]:
    """Pick a single ``(nfe, decode_method)`` for the unified pass.

    SCAPES runs one ODE solve over the whole timeline, so per-segment ``nfe``
    can't differ within a single render. Use the first interpolation segment's
    knobs if present (matches how interpolation-heavy timelines were already
    being tuned); otherwise fall back to the defaults.
    """
    for seg in request.segments:
        if seg.type == "interpolation":
            return seg.nfe, seg.decode_method
    return 8, "ola_smooth"


def render_timeline_audio(
    request: RenderRequest,
    *,
    cancel_event: Optional[threading.Event] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> bytes:
    """Render a clip / silence / interpolation timeline to a 16-bit PCM WAV.

    The whole timeline collapses into a single per-atom context schedule that
    SCAPES generates and decodes in one pass. There are no segment boundaries
    in the audio — the autoregressive past buffer carries timbre across what
    were previously seams. Silence segments still produce true zero samples:
    we generate atoms across them (to keep the past buffer warm for the next
    clip) and zero the corresponding sample ranges after decoding.

    Negative ``distance_sec`` (overlap) is handled at the schedule layer: the
    neighbor clip contributes fewer atoms, and the interpolation slot occupies
    the freed-up atoms — no sample-level trimming after generation.

    ``progress(done, total)`` ticks per-atom over the unified generation; the
    bar progresses smoothly across the whole timeline. ``cancel_event``, if
    set, raises ``RuntimeError`` within one atom (~0.3 s).
    """
    engine = get_inference_engine()
    sample_rate = engine.sr
    segs = request.segments

    generable = any(s.type in ("clip", "interpolation") for s in segs)
    if not generable:
        # Silence-only timeline: skip SCAPES entirely.
        tensors = [
            _make_silence_audio(s.duration, sample_rate=sample_rate)
            for s in segs
            if s.type == "silence"
        ]
        if not tensors:
            raise ValueError("timeline produced no audio")
        stitched = torch.cat(tensors, dim=-1)
        return _waveform_to_wav_bytes(stitched, sample_rate)

    schedule, silence_ranges = build_unified_timeline_schedule(request, engine)
    if not schedule:
        raise ValueError("timeline produced no atoms to generate")

    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("cancelled by client")

    timeline = engine.build_base_timeline(
        atoms_129D=[None] * len(schedule),
        context_embeddings=schedule,
        default_TF=False,
        default_AF=0.0,
    )

    nfe, decode_method = _resolve_render_params(request)
    start_time = time.time()
    timeline = run_generate_with_progress(
        engine,
        timeline,
        nfe=nfe,
        cancel_event=cancel_event,
        progress=progress,
    )
    elapsed = time.time() - start_time

    audio = engine.decode_timeline(timeline, method=decode_method)
    audio = audio.clamp(-1.0, 1.0).cpu()
    if audio.dim() == 3:
        audio = audio.squeeze(0)
    if audio.dim() != 2:
        raise ValueError(
            f"Expected decoded audio to be 2D [C, T], got shape {tuple(audio.shape)}"
        )

    total_samples = audio.shape[-1]
    for span in silence_ranges:
        start = max(0, min(total_samples, span.start_sample))
        end = max(start, min(total_samples, span.end_sample))
        if end > start:
            audio[..., start:end] = 0.0

    logger.info(
        "unified render: %d segments, %d atoms, %.3fs gen, %.3fs audio",
        len(segs),
        len(schedule),
        elapsed,
        total_samples / sample_rate,
    )
    return _waveform_to_wav_bytes(audio, sample_rate)


def render_interpolation_to_file(
    request: InterpolationElement,
    output_path: Path,
    *,
    cancel_event: Optional[threading.Event] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> int:
    """Render a single interpolation by routing through the unified timeline."""
    timeline_request = RenderRequest(
        segments=[InterpolationSegment.from_element(request)]
    )
    return render_timeline_to_file(
        timeline_request,
        output_path,
        cancel_event=cancel_event,
        progress=progress,
    )


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

