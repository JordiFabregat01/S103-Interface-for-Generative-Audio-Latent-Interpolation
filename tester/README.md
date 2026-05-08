# Timeline Tester

Standalone, throwaway GUI to exercise the refactored `/interpolate` endpoint
end-to-end. Lives outside `app/` and never touches the backend code.

```
+--------------------------------------------------+
| Parameter Panel                                  |
| audio1, audio2, duration_sec, stay_time_sec,     |
| stickyness, nfe, context_mode, decode_method     |
+--------------------------------------------------+
| Timeline (drag clip B horizontally)              |
|   [== A ====v====]   [== B ==v==]                |
|             ^anchor          ^anchor             |
|                  ↳ shaded interpolation zone     |
+--------------------------------------------------+
| [ Render ]   ▶ audio player    status: idle      |
+--------------------------------------------------+
```

## Prerequisites

- Node.js 18+ and npm (or pnpm/yarn). The project is set up for npm.
- The backend running on `http://localhost:8000`. From the repo root:
  ```bash
  cd app/backend
  python main.py
  ```

## Install + run

```bash
cd tester
npm install
npm run dev          # opens http://localhost:5174
```

The Vite dev server proxies `/api/*` to `http://localhost:8000/*`, so the
browser sees same-origin requests and CORS is a non-issue. The backend stays
unchanged.

## Parameter cheat sheet

| Control            | Maps to (`InterpolationElement`)                | Notes |
|--------------------|--------------------------------------------------|-------|
| `audio1` dropdown  | `audio1: AudioElement`                           | Limited to current backend enum: `camp_fire`, `keyboard`. |
| `audio2` dropdown  | `audio2: AudioElement`                           | Same. |
| Drag clip B        | `distance_sec`                                   | < 0 overlap, == 0 adjacent, > 0 gap. Snaps to 0.05s. |
| `duration_sec`     | `duration_sec` (only used when `distance_sec == 0`) | Disabled and shows `|distance_sec|` when non-zero. |
| Drag A anchor      | `a_anchor_sec`                                   | Scrubber maps clip width to `[0, 30s]` of source time. |
| Drag B anchor      | `b_anchor_sec`                                   | Same. |
| `stay_time_sec`    | `stay_time_sec`                                  | Pre-bridge dwell on A's context. |
| `stickyness`       | `stickyness`                                     | Sticky-curve sharpness. |
| `nfe`              | `nfe`                                            | ODE solver steps. |
| `context_mode`     | `context_mode`                                   | `auto` resolves from `distance_sec`. |
| `decode_method`    | `decode_method`                                  | `ola_smooth` is the SCAPES default. |

The "mode (resolved)" readout shows what `context_mode: auto` resolves to,
mirroring `request_from_clip_geometry` in
[../app/backend/inference/interpolation.py](../app/backend/inference/interpolation.py).

## Smoke tests

Run these in order with the backend up. Use the "Request payload" expander at
the bottom of the page to confirm the JSON being sent.

1. **Default state — overlap of -1.0s, both anchors 0.0.**
   Hit Render → expect a ~1-second WAV playing in `dynamic` mode.

2. **Drag clip B right until `distance_sec ≈ +2.0s`.**
   Hit Render → expect a ~2-second WAV in `static_at_anchor` mode.

3. **Drag clip B back to `distance_sec = 0` (adjacent), leave `duration_sec`
   blank in the panel.** Hit Render → expect a friendly 422 error in the
   status line ("`adjacent clips (distance_sec == 0) need a positive
   duration_sec`"). This validates the Pydantic guard.

4. **Same `distance_sec = 0`, set `duration_sec = 1.5`, force
   `context_mode = dynamic`.** Hit Render → expect a 1.5-second render that
   audibly differs from the `static_at_anchor` baseline.

For each render, the status line shows the elapsed time and WAV byte size, so
you also get a rough sense of perf as you change `nfe`.

## Out of scope

- Source-clip preview: there is no `<audio>` for the source `.wav` files
  themselves. That would require `/sounds/{id}` which lives in Omar's branch
  (not yet merged).
- Trim handles per clip. Anchor scrubbers cover most of that surface area.
- Multi-clip / N-transition timelines and stitched output.
- Production build, tests, lint config. This is a dev-only tool.

## How to wipe

```bash
rm -rf tester/
```

Nothing in `app/`, the backend, or the SCAPES submodule depends on this
folder.
