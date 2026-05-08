import { useCallback, useEffect, useRef, useState } from 'react'
import {
  InterpolationError,
  postInterpolate,
  revokeAudioUrl,
} from '../api'
import type { InterpolationRequest, RenderStatus } from '../types'

export interface RenderBarProps {
  request: InterpolationRequest
}

interface RenderState {
  status: RenderStatus
  audioUrl: string | null
  error: string | null
  elapsedMs: number | null
  byteLength: number | null
}

const INITIAL: RenderState = {
  status: 'idle',
  audioUrl: null,
  error: null,
  elapsedMs: null,
  byteLength: null,
}

export function RenderBar(props: RenderBarProps) {
  const { request } = props
  const [state, setState] = useState<RenderState>(INITIAL)
  // Keep a ref so unmount cleanup sees the latest URL even if it was set by a
  // late-completing request after a re-render.
  const audioUrlRef = useRef<string | null>(null)
  const inFlight = state.status === 'rendering'

  useEffect(() => {
    return () => {
      revokeAudioUrl(audioUrlRef.current)
      audioUrlRef.current = null
    }
  }, [])

  const handleRender = useCallback(async () => {
    setState((prev) => ({
      ...prev,
      status: 'rendering',
      error: null,
    }))
    const startedAt = performance.now()
    try {
      const res = await postInterpolate(request)
      revokeAudioUrl(audioUrlRef.current)
      audioUrlRef.current = res.audioUrl
      setState({
        status: 'done',
        audioUrl: res.audioUrl,
        error: null,
        elapsedMs: res.elapsedMs,
        byteLength: res.byteLength,
      })
    } catch (err) {
      const elapsedMs = performance.now() - startedAt
      const detail =
        err instanceof InterpolationError
          ? `${err.status}: ${err.detail}`
          : err instanceof Error
          ? err.message
          : String(err)
      setState((prev) => ({
        ...prev,
        status: 'error',
        error: detail,
        elapsedMs,
      }))
    }
  }, [request])

  return (
    <section className="panel render-bar" aria-label="Render">
      <button
        type="button"
        className="render-button"
        onClick={handleRender}
        disabled={inFlight}
      >
        {inFlight ? 'Rendering…' : 'Render'}
      </button>

      <div className="render-audio">
        {state.audioUrl ? (
          <audio
            controls
            src={state.audioUrl}
            // Reload when src changes; React handles this via the key.
            key={state.audioUrl}
          />
        ) : (
          <div className="render-audio-placeholder">no audio yet</div>
        )}
      </div>

      <StatusLine state={state} />
    </section>
  )
}

function StatusLine({ state }: { state: RenderState }) {
  switch (state.status) {
    case 'idle':
      return <div className="status idle">status: idle</div>
    case 'rendering':
      return <div className="status rendering">status: rendering…</div>
    case 'error':
      return (
        <div className="status error">
          <strong>error:</strong> {state.error ?? 'unknown'}
          {state.elapsedMs !== null
            ? ` (after ${(state.elapsedMs / 1000).toFixed(2)}s)`
            : ''}
        </div>
      )
    case 'done':
      return (
        <div className="status done">
          <strong>done</strong>
          {' · '}
          {state.elapsedMs !== null
            ? `${(state.elapsedMs / 1000).toFixed(2)}s`
            : ''}
          {state.byteLength !== null
            ? ` · ${(state.byteLength / 1024).toFixed(1)} KiB`
            : ''}
        </div>
      )
  }
}
