import { useCallback, type PointerEvent as ReactPointerEvent } from 'react'
import {
  resolveContextMode,
  resolveDurationSec,
  type AudioValue,
  type ContextMode,
} from '../types'
import { ClipBlock } from './ClipBlock'

export const PX_PER_SEC = 80
export const CLIP_VISUAL_WIDTH_SEC = 4
export const CLIP_VISUAL_WIDTH_PX = CLIP_VISUAL_WIDTH_SEC * PX_PER_SEC
export const TIMELINE_WIDTH_PX = 1000
export const SOURCE_MAX_SEC = 30
const DISTANCE_MIN = -(CLIP_VISUAL_WIDTH_SEC - 0.1)
const DISTANCE_MAX =
  (TIMELINE_WIDTH_PX - 2 * CLIP_VISUAL_WIDTH_PX) / PX_PER_SEC

export interface TimelinePatch {
  distance_sec?: number
  a_anchor_sec?: number
  b_anchor_sec?: number
}

export interface TimelineProps {
  audio1: AudioValue
  audio2: AudioValue
  distance_sec: number
  duration_sec: number | null
  a_anchor_sec: number
  b_anchor_sec: number
  context_mode: ContextMode
  onChange: (patch: TimelinePatch) => void
  disabled?: boolean
}

export function Timeline(props: TimelineProps) {
  const {
    audio1,
    audio2,
    distance_sec,
    duration_sec,
    a_anchor_sec,
    b_anchor_sec,
    context_mode,
    onChange,
    disabled = false,
  } = props

  const clipAX = 0
  const clipBX = CLIP_VISUAL_WIDTH_PX + distance_sec * PX_PER_SEC

  const resolvedMode = resolveContextMode(context_mode, distance_sec)
  const effectiveDuration = resolveDurationSec(distance_sec, duration_sec)

  const bridge = computeBridgeRect(
    distance_sec,
    duration_sec,
    clipAX + CLIP_VISUAL_WIDTH_PX,
  )

  const relationLabel = relationFor(distance_sec)

  const handleClipBPointerDown = useCallback(
    (e: ReactPointerEvent<HTMLDivElement>) => {
      if (disabled) return
      // Don't start a clip drag if the pointer was on the anchor handle.
      const target = e.target as HTMLElement
      if (target.closest('.clip-anchor-marker')) return

      const startClientX = e.clientX
      const startDistance = distance_sec
      const elem = e.currentTarget
      elem.setPointerCapture(e.pointerId)

      const onMove = (ev: PointerEvent) => {
        const dxPx = ev.clientX - startClientX
        const dxSec = dxPx / PX_PER_SEC
        const next = clamp(startDistance + dxSec, DISTANCE_MIN, DISTANCE_MAX)
        onChange({ distance_sec: roundTo(next, 0.05) })
      }
      const onUp = (ev: PointerEvent) => {
        elem.releasePointerCapture(ev.pointerId)
        elem.removeEventListener('pointermove', onMove)
        elem.removeEventListener('pointerup', onUp)
        elem.removeEventListener('pointercancel', onUp)
      }
      elem.addEventListener('pointermove', onMove)
      elem.addEventListener('pointerup', onUp)
      elem.addEventListener('pointercancel', onUp)
    },
    [disabled, distance_sec, onChange],
  )

  return (
    <section className="panel timeline-panel" aria-label="Timeline">
      <div className="panel-header">
        <span>Timeline</span>
        <span className={`relation-badge relation-${relationLabel}`}>
          {relationLabel}
        </span>
      </div>

      <div
        className="timeline-track"
        style={{ width: `${TIMELINE_WIDTH_PX}px` }}
      >
        {bridge ? (
          <div
            className={`bridge-zone bridge-${relationLabel}`}
            style={{ left: `${bridge.x}px`, width: `${bridge.width}px` }}
            aria-label={`Interpolation zone ${relationLabel}`}
          />
        ) : null}

        <ClipBlock
          label="A"
          source={audio1}
          x={clipAX}
          width={CLIP_VISUAL_WIDTH_PX}
          anchorSec={a_anchor_sec}
          sourceMaxSec={SOURCE_MAX_SEC}
          onAnchorChange={(sec) =>
            onChange({ a_anchor_sec: roundTo(sec, 0.01) })
          }
          disabled={disabled}
        />

        <ClipBlock
          label="B"
          source={audio2}
          x={clipBX}
          width={CLIP_VISUAL_WIDTH_PX}
          anchorSec={b_anchor_sec}
          sourceMaxSec={SOURCE_MAX_SEC}
          onAnchorChange={(sec) =>
            onChange({ b_anchor_sec: roundTo(sec, 0.01) })
          }
          onClipPointerDown={handleClipBPointerDown}
          draggable
          disabled={disabled}
        />
      </div>

      <div className="timeline-readouts">
        <ReadoutRow
          items={[
            ['distance_sec', `${signed(distance_sec)}s`],
            [
              'duration_sec',
              effectiveDuration === null
                ? '— (set value)'
                : `${effectiveDuration.toFixed(2)}s`,
            ],
            ['mode (override)', context_mode],
            ['mode (resolved)', resolvedMode],
          ]}
        />
        <ReadoutRow
          items={[
            ['a_anchor_sec', `${a_anchor_sec.toFixed(2)}s`],
            ['b_anchor_sec', `${b_anchor_sec.toFixed(2)}s`],
          ]}
        />
      </div>

      <p className="timeline-hint">
        Drag clip B horizontally to set <code>distance_sec</code>. Drag the small
        markers inside each clip to set <code>a_anchor_sec</code> /
        <code>b_anchor_sec</code>. Anchor scrubbers map the clip width to
        [0, {SOURCE_MAX_SEC}s] of source time; the backend clamps to the actual
        encoded length.
      </p>
    </section>
  )
}

function relationFor(distance_sec: number): 'overlap' | 'adjacent' | 'gap' {
  if (distance_sec < 0) return 'overlap'
  if (distance_sec > 0) return 'gap'
  return 'adjacent'
}

interface BridgeRect {
  x: number
  width: number
}

function computeBridgeRect(
  distance_sec: number,
  duration_sec: number | null,
  clipARightPx: number,
): BridgeRect | null {
  if (distance_sec > 0) {
    return {
      x: clipARightPx,
      width: distance_sec * PX_PER_SEC,
    }
  }
  if (distance_sec < 0) {
    const overlapPx = -distance_sec * PX_PER_SEC
    return {
      x: clipARightPx - overlapPx,
      width: overlapPx,
    }
  }
  if (duration_sec !== null && duration_sec > 0) {
    const widthPx = duration_sec * PX_PER_SEC
    return {
      x: clipARightPx - widthPx / 2,
      width: widthPx,
    }
  }
  return null
}

function ReadoutRow(props: { items: Array<[string, string]> }) {
  return (
    <div className="readout-row">
      {props.items.map(([label, value]) => (
        <div key={label} className="readout">
          <span className="readout-label">{label}</span>
          <span className="readout-value">{value}</span>
        </div>
      ))}
    </div>
  )
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v))
}

function roundTo(v: number, step: number): number {
  return Math.round(v / step) * step
}

function signed(v: number): string {
  if (v > 0) return `+${v.toFixed(2)}`
  return v.toFixed(2)
}
