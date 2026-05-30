"""Clip-aware interpolation API for the timeline app.

Pure Python module: no file I/O, no notebook side effects, no print/tqdm/display.
Wraps SCAPES `FlowInference` via its public methods only; the submodule is
never modified from here.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

import torch
from torch import Tensor

from inference.scapes_runtime import (
    FlowInference,
    low_pass_filter,
    slerp,
    sticky_curve_torch,
)

logger = logging.getLogger(__name__)


ContextMode = Literal["auto", "static_first", "static_at_anchor", "dynamic"]
ConcreteContextMode = Literal["static_first", "static_at_anchor", "dynamic"]

_CONCRETE_CONTEXT_MODES: tuple[str, ...] = (
    "static_first",
    "static_at_anchor",
    "dynamic",
)

# Decode methods that use the asymmetric prefix-padding OLA window. Their first
# `macro_overlap_samples` are guaranteed-silent prefix (no past audio feeds the
# zero-padded region), and the last `crossfade_samples` taper to zero (the
# trailing hann fade of the final atom). For interpolation we trim both edges.
_OLA_DECODE_METHODS: frozenset[str] = frozenset({"ola", "ola_smooth"})


@dataclass
class EncodedSource:
    """A source audio file already encoded by SCAPES.

    `contexts` are the only thing pure interpolation needs (slerp consumes them).
    `atoms` is reserved for callers that want to mix in the source's own audio
    via `AF > 0`; for the interpolation path it can stay None.
    """

    source_id: str
    contexts: list[Tensor]
    atoms: Optional[list[Tensor]] = None
    sample_rate: int = 48000


@dataclass
class InterpolationRequest:
    """A geometry-resolved request handed to `interpolate()`.

    `context_mode` may be `"auto"` here for serialization symmetry, but
    `request_from_clip_geometry` always returns a concrete mode. `interpolate()`
    will resolve any leftover `"auto"` to `"static_at_anchor"` with a log line.
    """

    source_a: EncodedSource
    source_b: EncodedSource
    duration_sec: float
    a_anchor_sec: float = 0.0
    b_anchor_sec: float = 0.0
    context_mode: ContextMode = "auto"
    stay_time_sec: float = 0.0
    stickyness: float = 1.0
    nfe: int = 8
    decode_method: str = "ola_smooth"


@dataclass
class InterpolationResult:
    """The output of `interpolate()`."""

    audio: Tensor
    atoms_generated: list[Tensor] = field(default_factory=list)
    sample_rate: int = 48000
    timeline_size: int = 0
    duration_sec: float = 0.0
    context_mode: ConcreteContextMode = "static_at_anchor"


class InterpolationCancelled(Exception):
    """Raised when `cancel_event` was set before / after generation."""


def _hop_seconds(engine: FlowInference) -> float:
    return engine.hop_samples / engine.sr


def _seconds_to_atoms(sec: float, hop_sec: float) -> int:
    return int(round(float(sec) / hop_sec))


def _clamp_anchor_atom(anchor_sec: float, hop_sec: float, num_contexts: int) -> int:
    if num_contexts <= 0:
        raise ValueError("source has no contexts")
    raw = _seconds_to_atoms(anchor_sec, hop_sec)
    return max(0, min(num_contexts - 1, raw))


def _resolve_context_mode(mode: ContextMode) -> ConcreteContextMode:
    if mode == "auto":
        logger.info(
            "context_mode='auto' resolved to 'static_at_anchor' "
            "(no geometry available at interpolate() boundary)"
        )
        return "static_at_anchor"
    if mode not in _CONCRETE_CONTEXT_MODES:
        raise ValueError(
            f"invalid context_mode: {mode!r}; expected one of "
            f"'auto', {_CONCRETE_CONTEXT_MODES}"
        )
    return mode  # type: ignore[return-value]


def _dynamic_window(
    contexts: list[Tensor],
    anchor_atom: int,
    timeline_size: int,
    *,
    source_id: str,
) -> list[Tensor]:
    """Slice `contexts[anchor:anchor+N]`, padding with the last context if needed."""
    if len(contexts) == 0:
        raise ValueError(f"source {source_id!r} has no contexts")

    end = anchor_atom + timeline_size
    if anchor_atom >= len(contexts):
        logger.warning(
            "dynamic window for source %r starts past last context "
            "(anchor_atom=%d, num_contexts=%d); padding entirely with last context",
            source_id,
            anchor_atom,
            len(contexts),
        )
        return [contexts[-1]] * timeline_size

    window = list(contexts[anchor_atom:end])
    if len(window) < timeline_size:
        missing = timeline_size - len(window)
        logger.warning(
            "dynamic window for source %r is %d contexts short of timeline_size=%d; "
            "padding with last available context",
            source_id,
            missing,
            timeline_size,
        )
        window.extend([contexts[-1]] * missing)
    return window


def _build_alpha_curve(timeline_size: int, stay_time: int, stickyness: float) -> Tensor:
    if stickyness <= 0:
        raise ValueError("stickyness must be > 0")
    if stay_time < 0:
        raise ValueError("stay_time must be non-negative")
    inner = timeline_size - 2 * stay_time
    if inner < 1:
        raise ValueError(
            f"stay_time too large: timeline_size={timeline_size}, "
            f"stay_time={stay_time} (needs timeline_size - 2*stay_time >= 1)"
        )
    if inner == 1:
        # sticky_curve_torch with n_points=1 is degenerate (linspace gives [0]);
        # use a single midpoint so the slerp produces a true mid-blend.
        ramp = torch.tensor([0.5])
    else:
        ramp = sticky_curve_torch(n_points=inner, stickiness=stickyness)
    full = torch.cat([torch.zeros(stay_time), ramp, torch.ones(stay_time)])
    smoothed = low_pass_filter(full, alpha=0.5)
    # `low_pass_filter` is 10 causal EMA passes with `filtered[0]` pinned to 0;
    # the accumulated lag means `smoothed[-1]` never reaches 1 on practical
    # timelines (e.g. ~0.66 for N=30), so the slerp stops short of source B
    # and the interpolation feels biased toward source A. Renormalize so the
    # curve always spans the full [0, 1] range end-to-end.
    if smoothed.numel() > 1:
        span = smoothed[-1] - smoothed[0]
        if span.abs() > 1e-6:
            smoothed = (smoothed - smoothed[0]) / span
    return smoothed


def _check_cancelled(cancel_event: Optional[threading.Event]) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise InterpolationCancelled("interpolation cancelled by caller")


def _trim_ola_padding(audio: Tensor, engine: FlowInference) -> Tensor:
    """Strip the asymmetric-OLA silent prefix and trailing fade from a render.

    SCAPES' OLA window is `[zeros | hann_in | ones | hann_out]` with a
    `macro_overlap - crossfade` frame zero prefix. For atoms 1..N-1 those
    zeros overlap with the previous atom's audible region and reconstruction
    works; for atom 0 there is no previous atom, so the first
    `macro_overlap_samples` of the rendered buffer are guaranteed silence.
    Symmetrically, the last atom's trailing `crossfade_samples` are a Hann
    fade to zero. Trim both so the interpolation hands off at full level to
    the surrounding clip rather than through a silent gap.
    """
    head = int(getattr(engine, "macro_overlap_samples", 0))
    tail = int(getattr(engine, "crossfade_samples", 0))
    if head < 0:
        head = 0
    if tail < 0:
        tail = 0
    total = audio.shape[-1]
    if total <= head + tail:
        # Pathological short render; leave it alone rather than emit zero-length.
        return audio
    return audio[..., head: total - tail]


def interpolate(
    engine: FlowInference,
    request: InterpolationRequest,
    *,
    cancel_event: Optional[threading.Event] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> InterpolationResult:
    """Run a SCAPES interpolation between two pre-encoded sources.

    Coarse-grained cancellation: `cancel_event` is checked once before and
    once after `engine.generate()`. Fine-grained per-step cancellation is a
    deliberate follow-up.

    `progress(done, total)` fires with `(0, N)` and `(N, N)` around generation.
    """
    if request.duration_sec <= 0:
        raise ValueError(f"duration_sec must be > 0 (got {request.duration_sec})")

    hop_sec = _hop_seconds(engine)
    timeline_size = max(1, _seconds_to_atoms(request.duration_sec, hop_sec))
    stay_time = _seconds_to_atoms(request.stay_time_sec, hop_sec)

    mode = _resolve_context_mode(request.context_mode)

    contexts_a = request.source_a.contexts
    contexts_b = request.source_b.contexts
    if not contexts_a or not contexts_b:
        raise ValueError("both sources must have at least one context embedding")

    alpha = _build_alpha_curve(timeline_size, stay_time, request.stickyness).to(
        engine.device
    )

    if mode == "static_first":
        c_a_window = [contexts_a[0]] * timeline_size
        c_b_window = [contexts_b[0]] * timeline_size
    elif mode == "static_at_anchor":
        a_idx = _clamp_anchor_atom(request.a_anchor_sec, hop_sec, len(contexts_a))
        b_idx = _clamp_anchor_atom(request.b_anchor_sec, hop_sec, len(contexts_b))
        c_a_window = [contexts_a[a_idx]] * timeline_size
        c_b_window = [contexts_b[b_idx]] * timeline_size
    else:  # "dynamic"
        a_anchor_atom = max(0, _seconds_to_atoms(request.a_anchor_sec, hop_sec))
        b_anchor_atom = max(0, _seconds_to_atoms(request.b_anchor_sec, hop_sec))
        c_a_window = _dynamic_window(
            contexts_a, a_anchor_atom, timeline_size, source_id=request.source_a.source_id
        )
        c_b_window = _dynamic_window(
            contexts_b, b_anchor_atom, timeline_size, source_id=request.source_b.source_id
        )

    contexts: list[Tensor] = []
    for t in range(timeline_size):
        c_a_t = c_a_window[t].to(engine.device)
        c_b_t = c_b_window[t].to(engine.device)
        contexts.append(slerp(c_a_t, c_b_t, alpha[t]))

    timeline = engine.build_base_timeline(
        atoms_129D=[None] * timeline_size,
        context_embeddings=contexts,
        default_TF=False,
        default_AF=0.0,
    )

    _check_cancelled(cancel_event)
    if progress is not None:
        progress(0, timeline_size)

    timeline = engine.generate(timeline, NFE=request.nfe)

    _check_cancelled(cancel_event)
    if progress is not None:
        progress(timeline_size, timeline_size)

    audio = engine.decode_timeline(timeline, method=request.decode_method)
    audio = audio.clamp(-1.0, 1.0).cpu()
    if audio.dim() == 3:
        audio = audio.squeeze(0)

    if request.decode_method in _OLA_DECODE_METHODS:
        audio = _trim_ola_padding(audio, engine)

    atoms_generated = [
        step["atom_generated"].detach().cpu()
        for step in timeline
        if step.get("atom_generated") is not None
    ]

    actual_duration = audio.shape[-1] / engine.sr

    logger.info(
        "interpolate: timeline_size=%d, context_mode=%s, nfe=%d, duration_sec=%.3f",
        timeline_size,
        mode,
        request.nfe,
        actual_duration,
    )

    return InterpolationResult(
        audio=audio,
        atoms_generated=atoms_generated,
        sample_rate=engine.sr,
        timeline_size=timeline_size,
        duration_sec=actual_duration,
        context_mode=mode,
    )


def request_from_clip_geometry(
    source_a: EncodedSource,
    source_b: EncodedSource,
    distance_sec: float,
    *,
    adjacent_duration_sec: Optional[float] = None,
    a_anchor_sec: float = 0.0,
    b_anchor_sec: float = 0.0,
    stay_time_sec: float = 0.0,
    stickyness: float = 1.0,
    nfe: int = 8,
    decode_method: str = "ola_smooth",
    context_mode_override: ContextMode = "auto",
) -> InterpolationRequest:
    """Map clip-pair geometry to a concrete `InterpolationRequest`.

    `distance_sec`:
      * `< 0` -> overlap of `|distance_sec|`. Default mode: dynamic.
      * `> 0` -> gap of `distance_sec`. Default mode: static_at_anchor.
      * `== 0` -> adjacent. Requires `adjacent_duration_sec`. Mode: static_at_anchor.

    `context_mode_override` of anything other than `"auto"` wins over the
    geometry choice.
    """
    if distance_sec < 0:
        duration = -distance_sec
        geom_mode: ConcreteContextMode = "dynamic"
    elif distance_sec > 0:
        duration = distance_sec
        geom_mode = "static_at_anchor"
    else:
        if adjacent_duration_sec is None or adjacent_duration_sec <= 0:
            raise ValueError(
                "adjacent clips need an explicit positive `adjacent_duration_sec`"
            )
        duration = float(adjacent_duration_sec)
        geom_mode = "static_at_anchor"

    if context_mode_override == "auto":
        mode: ContextMode = geom_mode
    elif context_mode_override in _CONCRETE_CONTEXT_MODES:
        mode = context_mode_override
    else:
        raise ValueError(
            f"invalid context_mode_override: {context_mode_override!r}"
        )

    return InterpolationRequest(
        source_a=source_a,
        source_b=source_b,
        duration_sec=duration,
        a_anchor_sec=a_anchor_sec,
        b_anchor_sec=b_anchor_sec,
        context_mode=mode,
        stay_time_sec=stay_time_sec,
        stickyness=stickyness,
        nfe=nfe,
        decode_method=decode_method,
    )


def interpolate_clips(
    engine: FlowInference,
    source_a: EncodedSource,
    source_b: EncodedSource,
    distance_sec: float,
    *,
    adjacent_duration_sec: Optional[float] = None,
    a_anchor_sec: float = 0.0,
    b_anchor_sec: float = 0.0,
    stay_time_sec: float = 0.0,
    stickyness: float = 1.0,
    nfe: int = 8,
    decode_method: str = "ola_smooth",
    context_mode_override: ContextMode = "auto",
    cancel_event: Optional[threading.Event] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> InterpolationResult:
    """Convenience: derive a request from clip geometry, then interpolate."""
    request = request_from_clip_geometry(
        source_a,
        source_b,
        distance_sec,
        adjacent_duration_sec=adjacent_duration_sec,
        a_anchor_sec=a_anchor_sec,
        b_anchor_sec=b_anchor_sec,
        stay_time_sec=stay_time_sec,
        stickyness=stickyness,
        nfe=nfe,
        decode_method=decode_method,
        context_mode_override=context_mode_override,
    )
    return interpolate(
        engine,
        request,
        cancel_event=cancel_event,
        progress=progress,
    )
