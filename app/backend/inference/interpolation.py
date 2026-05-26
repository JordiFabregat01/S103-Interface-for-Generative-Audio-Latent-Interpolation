"""Clip-aware interpolation API for the timeline app.

Pure Python module: no file I/O, no notebook side effects, no print/tqdm/display.
Wraps SCAPES `FlowInference` via its public methods only; the submodule is
never modified from here.
"""

from __future__ import annotations

import logging
import math
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

# Overlap / dynamic bridges need enough ODE steps for an audible morph.
MIN_TIMELINE_SIZE_DYNAMIC = 5


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


def _segment_seconds(engine: FlowInference) -> float:
    """Length of one decoded atom window (matches ``decode_timeline`` geometry)."""
    return engine.segment_samples / engine.sr


def _decoded_duration_sec(engine: FlowInference, timeline_size: int) -> float:
    """Audio length SCAPES emits for ``timeline_size`` OLA steps."""
    if timeline_size < 1:
        return 0.0
    hop_sec = _hop_seconds(engine)
    segment_sec = _segment_seconds(engine)
    return (timeline_size - 1) * hop_sec + segment_sec


def _timeline_size_for_duration(engine: FlowInference, duration_sec: float) -> int:
    """Pick ``timeline_size`` so decoded audio is at least ``duration_sec``.

    SCAPES decodes with ``(N-1)*hop_samples + segment_samples``, not ``N*hop``.
    """
    hop_sec = _hop_seconds(engine)
    segment_sec = _segment_seconds(engine)
    if duration_sec <= segment_sec:
        return 1
    hops_needed = (float(duration_sec) - segment_sec) / hop_sec
    return max(1, int(math.ceil(hops_needed)) + 1)


def _seconds_to_atoms(sec: float, hop_sec: float) -> int:
    return int(round(float(sec) / hop_sec))


def validate_overlap_anchors(
    distance_sec: float,
    a_anchor_sec: float,
    b_anchor_sec: float,
    *,
    allow_zero_anchors: bool = False,
) -> None:
    """Overlap must not use dynamic mode with both anchors at 0 unless timeline sets them."""
    if distance_sec >= 0 or allow_zero_anchors:
        return
    if a_anchor_sec == 0.0 and b_anchor_sec == 0.0:
        raise ValueError(
            "overlap (distance_sec < 0) requires non-default anchors "
            "(a_anchor_sec / b_anchor_sec) or a full /render timeline with flanking clips"
        )


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
    """Slice ``contexts[anchor:anchor+N]`` without tail padding (caller caps ``N``)."""
    if len(contexts) == 0:
        raise ValueError(f"source {source_id!r} has no contexts")

    if anchor_atom >= len(contexts):
        raise ValueError(
            f"dynamic window for source {source_id!r} starts past last context "
            f"(anchor_atom={anchor_atom}, num_contexts={len(contexts)})"
        )

    end = anchor_atom + timeline_size
    window = list(contexts[anchor_atom:end])
    if len(window) < timeline_size:
        raise ValueError(
            f"dynamic window for source {source_id!r} needs {timeline_size} contexts "
            f"from anchor {anchor_atom}, but only {len(window)} are available "
            f"(num_contexts={len(contexts)})"
        )
    return window


def _cap_timeline_for_dynamic(
    timeline_size: int,
    contexts_a: list[Tensor],
    contexts_b: list[Tensor],
    a_anchor_atom: int,
    b_anchor_atom: int,
) -> int:
    """Limit ``timeline_size`` to contexts available after each anchor."""
    avail_a = len(contexts_a) - a_anchor_atom
    avail_b = len(contexts_b) - b_anchor_atom
    if avail_a <= 0 or avail_b <= 0:
        raise ValueError(
            f"dynamic interpolation: no contexts after anchors "
            f"(a_anchor_atom={a_anchor_atom}, len_a={len(contexts_a)}, "
            f"b_anchor_atom={b_anchor_atom}, len_b={len(contexts_b)})"
        )
    capped = min(timeline_size, avail_a, avail_b)
    if capped < timeline_size:
        logger.warning(
            "capping timeline_size from %d to %d (contexts available after anchors)",
            timeline_size,
            capped,
        )
    return max(1, capped)


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
    return low_pass_filter(full, alpha=0.5)


def _check_cancelled(cancel_event: Optional[threading.Event]) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise InterpolationCancelled("interpolation cancelled by caller")


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
    mode = _resolve_context_mode(request.context_mode)

    timeline_size = _timeline_size_for_duration(engine, request.duration_sec)
    stay_time = _seconds_to_atoms(request.stay_time_sec, hop_sec)
    if 2 * stay_time >= timeline_size:
        stay_time = max(0, (timeline_size - 1) // 2)

    contexts_a = request.source_a.contexts
    contexts_b = request.source_b.contexts
    if not contexts_a or not contexts_b:
        raise ValueError("both sources must have at least one context embedding")

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
        timeline_size = _cap_timeline_for_dynamic(
            max(timeline_size, MIN_TIMELINE_SIZE_DYNAMIC),
            contexts_a,
            contexts_b,
            a_anchor_atom,
            b_anchor_atom,
        )
        c_a_window = _dynamic_window(
            contexts_a, a_anchor_atom, timeline_size, source_id=request.source_a.source_id
        )
        c_b_window = _dynamic_window(
            contexts_b, b_anchor_atom, timeline_size, source_id=request.source_b.source_id
        )

    alpha = _build_alpha_curve(timeline_size, stay_time, request.stickyness).to(
        engine.device
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

    atoms_generated = [
        step["atom_generated"].detach().cpu()
        for step in timeline
        if step.get("atom_generated") is not None
    ]

    if audio.dim() == 2:
        actual_duration = audio.shape[-1] / engine.sr
    else:
        actual_duration = _decoded_duration_sec(engine, timeline_size)

    logger.info(
        "interpolate: timeline_size=%d, context_mode=%s, nfe=%d, "
        "requested_duration_sec=%.3f, decoded_duration_sec=%.3f",
        timeline_size,
        mode,
        request.nfe,
        request.duration_sec,
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

    validate_overlap_anchors(
        distance_sec,
        a_anchor_sec,
        b_anchor_sec,
        allow_zero_anchors=False,
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
