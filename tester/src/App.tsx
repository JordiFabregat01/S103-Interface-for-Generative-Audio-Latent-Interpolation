import { useCallback, useMemo, useState } from 'react'
import { ParamPanel, type ParamPatch } from './components/ParamPanel'
import { RenderBar } from './components/RenderBar'
import { Timeline, type TimelinePatch } from './components/Timeline'
import type { InterpolationRequest } from './types'

interface AppState {
  audio1: InterpolationRequest['audio1']
  audio2: InterpolationRequest['audio2']
  distance_sec: number
  /** User-provided value used only when distance_sec === 0. */
  duration_sec: number | null
  a_anchor_sec: number
  b_anchor_sec: number
  stay_time_sec: number
  stickyness: number
  nfe: number
  context_mode: InterpolationRequest['context_mode']
  decode_method: InterpolationRequest['decode_method']
}

const INITIAL: AppState = {
  audio1: 'camp_fire',
  audio2: 'keyboard',
  distance_sec: -1.0,
  duration_sec: 1.0,
  a_anchor_sec: 0.0,
  b_anchor_sec: 0.0,
  stay_time_sec: 0.0,
  stickyness: 1.0,
  nfe: 8,
  context_mode: 'auto',
  decode_method: 'ola_smooth',
}

function App() {
  const [state, setState] = useState<AppState>(INITIAL)

  const onParamChange = useCallback(
    (patch: ParamPatch) => setState((prev) => ({ ...prev, ...patch })),
    [],
  )

  const onTimelineChange = useCallback(
    (patch: TimelinePatch) => setState((prev) => ({ ...prev, ...patch })),
    [],
  )

  const request = useMemo<InterpolationRequest>(
    () => ({
      audio1: state.audio1,
      audio2: state.audio2,
      distance_sec: state.distance_sec,
      // Backend ignores duration_sec when distance_sec !== 0; null it out so
      // the wire format reflects the active semantics.
      duration_sec:
        state.distance_sec === 0 ? state.duration_sec : null,
      a_anchor_sec: state.a_anchor_sec,
      b_anchor_sec: state.b_anchor_sec,
      stay_time_sec: state.stay_time_sec,
      stickyness: state.stickyness,
      nfe: state.nfe,
      context_mode: state.context_mode,
      decode_method: state.decode_method,
    }),
    [state],
  )

  return (
    <div className="app">
      <header className="app-header">
        <h1>SCAPES interpolation tester</h1>
        <p className="app-tagline">
          Drag clip B to set <code>distance_sec</code>. Drag the anchor markers
          to set <code>a_anchor_sec</code> / <code>b_anchor_sec</code>. Hit
          Render to POST <code>/interpolate</code> and play the bridge.
        </p>
      </header>

      <ParamPanel
        audio1={state.audio1}
        audio2={state.audio2}
        duration_sec={state.duration_sec}
        stay_time_sec={state.stay_time_sec}
        stickyness={state.stickyness}
        nfe={state.nfe}
        context_mode={state.context_mode}
        decode_method={state.decode_method}
        distance_sec={state.distance_sec}
        onChange={onParamChange}
      />

      <Timeline
        audio1={state.audio1}
        audio2={state.audio2}
        distance_sec={state.distance_sec}
        duration_sec={state.duration_sec}
        a_anchor_sec={state.a_anchor_sec}
        b_anchor_sec={state.b_anchor_sec}
        context_mode={state.context_mode}
        onChange={onTimelineChange}
      />

      <RenderBar request={request} />

      <details className="request-preview">
        <summary>Request payload</summary>
        <pre>{JSON.stringify(request, null, 2)}</pre>
      </details>
    </div>
  )
}

export default App
