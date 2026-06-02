# Unified-timeline SCAPES rendering

## TL;DR

The backend used to render a timeline by running SCAPES once per segment and
concatenating the resulting WAVs. It now runs SCAPES **once** over the entire
user timeline. Clips, interpolations, and silences all live in a single flat
sequence of per-atom context embeddings that the model walks autoregressively
in one pass. The audio comes out without per-segment seams; silence regions are
zero-filled after decoding.

## Why we changed it

The previous renderer (`render_timeline_audio` before this refactor) looped over
the segments of a `RenderRequest`, called `engine.generate()` on each one
independently, decoded each one to audio, and `torch.cat`'d the results. That
shape was easy to reason about, but it had two real costs:

1. **Per-segment cold start.** SCAPES is autoregressive — every atom it
   generates conditions on the last `segment_length=5` atoms it produced. The
   `FlowInference.generate` loop initializes that buffer with `dummy_atom`
   (`modules/scapes/SCAPES/inference/FlowInference.py:470`), so the first ~5
   atoms of each independent call are produced without any real history. With
   N segments, that's ~5N atoms of timbre warm-up smear, and the smear sits
   exactly on the boundaries the user can hear.

2. **Sample-level seam bookkeeping.** Handling overlap (`distance_sec < 0`)
   required head/tail trimming in samples (`head_trim_samples`,
   `tail_trim_samples`), anchor overrides via `model_copy`, and a `torch.cat`
   at the end. None of that was wrong, but it was working around the fact that
   we were generating independent pieces and gluing them.

Once you notice that SCAPES is autoregressive *at the atom level* and the
`context_embedding` is a per-atom argument, the whole stitching layer becomes
unnecessary. You can describe the entire user timeline as one long list of
per-atom contexts and let the model run.

## How SCAPES actually works (the load-bearing insight)

`FlowInference.build_base_timeline(atoms_129D, context_embeddings, ...)` does
not know about segments. It takes two flat lists of the same length:

- `atoms_129D[t]` — an optional "given" atom for teacher-forcing slot `t`
- `context_embeddings[t]` — the CLAP-like semantic context vector for slot `t`

`FlowInference.generate(timeline, NFE)` then walks `t = 0, 1, …, N-1`. At each
step it:

- pulls the previous `M = segment_length` (5) atoms from the timeline as
  autoregressive history (or `dummy_atom` when `t < M`),
- encodes that history with the local encoder,
- conditions on `timeline[t]["context_embedding"]`,
- runs `model.generate(x0, encoded_past, context, max_nfe=NFE)` once,
- writes the result back into `timeline[t]["atom_generated"]`.

Nothing in that loop changes between "clip atom" and "interpolation atom" — the
only difference is what context vector lives at that index. So if we build a
context schedule like

    [ clip-A's contexts ]  [ slerp(A, B) curve ]  [ clip-B's contexts ]
    \________________ N atoms total ________________/

and hand it to `generate`, the past-atom buffer carries the timbre from
clip A's last atoms across the interpolation slot and into the start of
clip B — no cold start, no concat seam.

## The new pipeline

```
RenderRequest ─┐
               │
               ▼
build_unified_timeline_schedule(request, engine)
               │
               │  for each segment, append per-atom contexts:
               │   • clip → build_clip_context_schedule(source, dur, engine)
               │   • interpolation → build_interpolation_context_schedule(...)
               │   • silence → split prev/next clip context, record mute range
               │
               ▼
       (schedule, silence_ranges)
               │
               ▼
engine.build_base_timeline([None]*N, schedule)   ← one flat timeline
               │
               ▼
run_generate_with_progress(engine, timeline, ...)  ← one autoregressive pass
               │
               ▼
engine.decode_timeline(timeline, method="ola_smooth")  ← one global OLA
               │
               ▼
zero-fill silence sample ranges
               │
               ▼
       16-bit PCM WAV
```

Every step that used to be "do this N times then `torch.cat`" is now "do this
once over the whole schedule." The audio that comes out of `decode_timeline`
is already continuous; we don't reassemble it.

## Where things live now

- **`app/backend/inference/interpolation.py`**
  - `build_interpolation_context_schedule(engine, request)` — *new*. Returns
    the per-atom context list for an interpolation (the slerp curve over
    `static_first` / `static_at_anchor` / `dynamic` windows) without running
    any generation.
  - `interpolate(...)` — unchanged externally, but now thin: builds the
    schedule via the helper, then generates and decodes. Kept for the legacy
    single-interpolation API.
  - `request_from_clip_geometry`, `interpolate_clips` — unchanged.

- **`app/backend/inference/methods.py`**
  - `build_clip_context_schedule(source, duration_sec, engine)` — *new*. Slices
    a clip's pre-computed `EncodedSource.contexts` to the requested duration
    (pads with the last context if the request is longer than the source).
  - `_neighbor_clip_context(segs, idx, step, engine)` — *new*. For silence
    slots, finds the nearest clip or interpolation endpoint to inherit a
    context vector from.
  - `_interpolation_anchor_override(segs, idx, hop_sec)` — *new*. Computes the
    anchor overrides and per-neighbor clip atom-trims for one interpolation
    segment. Identical math to the old per-segment override code, but in atom
    units so trimming happens before generation, not after.
  - `build_unified_timeline_schedule(request, engine)` — *new*. The whole
    schedule builder. Returns `(schedule, silence_ranges)`.
  - `run_generate_with_progress(engine, timeline, nfe, cancel_event, progress)`
    — *new*. A line-for-line replica of `FlowInference.generate`'s loop with
    two hooks added inside the per-atom step: `progress(done, total)` after
    each atom, and `cancel_event` checked at the top of each step. The SCAPES
    submodule itself is not modified.
  - `render_timeline_audio(request, …)` — rewritten. Schedule → base timeline
    → unified generate → decode → mute silences → WAV.
  - `render_interpolation_audio(request)` and `render_interpolation_to_file(…)`
    — now one-liners that wrap their `InterpolationElement` in a
    single-segment `RenderRequest` and route through the unified path.

- **`app/backend/inference/models.py`**
  - `InterpolationSegment.from_element(element)` — *new*. Symmetric to the
    existing `to_element`; used by the legacy wrappers above.

## Silence handling

A silence segment is always rendered as **true silence**: the corresponding
sample range is hard-zeroed after decoding. SCAPES still *generates* atoms
across the gap — we never skip them — because the autoregressive past buffer
those atoms produce is what the next clip starts from. The only decision is
which context vector to feed those (muted) atoms, and that's chosen by the
silence's neighbors:

| neighbors                    | context fed across the silence atoms                       |
|------------------------------|------------------------------------------------------------|
| preceding **and** following  | first half = preceding sound's last context, second half = following sound's first context |
| preceding only (trailing gap)| preceding sound's last context, repeated                   |
| following only (leading gap) | following sound's first context, repeated                  |

The split for a between-clips gap is the load-bearing part. If we warmed the
whole silence with the preceding sound's context, the AR buffer entering the
next clip would still be full of the *previous* timbre — and because generation
has variance, that leaks an audible remnant of the previous sound into the start
of the next one once its volume comes back up. By handing the second half of the
gap the following sound's context, the next clip's AR history is already its own
timbre, so it starts clean. The samples in between are discarded either way.

Mechanics:
- The silence segment contributes `round(duration / hop_sec)` atoms. For a
  two-sided gap the first `atom_count // 2` carry the preceding context and the
  rest carry the following context (`build_unified_timeline_schedule`).
- Neighbor contexts come from `_neighbor_clip_context`, which walks to the
  nearest clip (or interpolation endpoint) on each side: `-1` returns that
  sound's *last* context, `+1` its *first*.
- We record `(start_sample, end_sample)` and zero those ranges after
  `decode_timeline`. The OLA crossfade windows handle the boundary samples.
- Trade-off: long silences cost compute (atoms whose samples are discarded),
  but no cold-start smear and no cross-boundary timbre bleed.

Silence-only timelines (no clips, no interpolations) short-circuit SCAPES
entirely and emit zeros directly via `_make_silence_audio`.

## Overlap handling (`distance_sec < 0`)

The old code generated full clip audios and then trimmed `overlap_samples` off
the neighboring clip ends. The new code does the trim at the atom-count layer:
when an interpolation segment has negative `distance_sec`, the previous and
next clip segments contribute `floor(duration/hop) - overlap_atoms` atoms each,
and the interpolation segment's atoms naturally occupy the freed-up slots.
There is no sample-level trimming after generation — the schedule already
describes the right composition.

## Progress and cancellation

Because the whole timeline now runs through one `generate` call, segment-level
progress (the old "1/3, 2/3, 3/3" ticks) doesn't fit anymore. Instead,
`run_generate_with_progress` fires `progress(done, total)` per atom, where
`total = len(schedule)`. That makes the progress bar advance smoothly across
the entire render, and cancellation now takes effect within one atom
(~0.3 s) instead of waiting for the next segment boundary.

The job runner in `app/backend/main.py` is unchanged — it already passes a
`progress` callback into `render_timeline_to_file`; only the granularity of
that callback has changed.

## What stayed the same

- **The HTTP API.** `POST /render`, `POST /interpolate` (deprecated),
  `POST /render/async`, `GET /jobs/{id}`, and `GET /jobs/{id}/result.wav` all
  accept and return the same payloads.
- **`RenderRequest` and segment schemas.** `ClipSegment`, `SilenceSegment`,
  `InterpolationSegment`, and `InterpolationElement` are unchanged on the wire.
  `InterpolationSegment` only gained a `from_element` classmethod, which is
  additive.
- **The SCAPES submodule.** Not modified. `run_generate_with_progress` is a
  parallel implementation that lives in the backend, by the same rule that
  governs `interpolation.py`: wrap SCAPES via its public methods, don't fork it.
- **Per-interpolation knobs.** `nfe`, `decode_method`, `context_mode`,
  `stay_time_sec`, `stickyness`, `a_anchor_sec`, `b_anchor_sec` all still
  shape the interpolation's slerp curve via `build_interpolation_context_schedule`.

The one subtlety: `nfe` and `decode_method` are *render-wide* in the unified
pass, because there's a single ODE solve and a single decode. We resolve them
from the first interpolation segment in the request (`_resolve_render_params`),
falling back to `nfe=8` / `ola_smooth` if the timeline has no interpolations.
Mixing different `nfe`s across interpolations in one render is no longer
supported — in practice the previous code's per-segment `nfe` choices were
rarely heterogeneous, and this matches how SCAPES actually generates.
