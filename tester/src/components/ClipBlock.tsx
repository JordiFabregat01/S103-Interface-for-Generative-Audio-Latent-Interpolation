import { useCallback, useRef, type PointerEvent as ReactPointerEvent } from 'react'
import type { AudioValue } from '../types'

export interface ClipBlockProps {
  label: string
  source: AudioValue
  /** Pixel position of the clip's left edge inside the timeline track. */
  x: number
  /** Pixel width of the clip block. */
  width: number
  /** Current anchor position in source time (seconds). */
  anchorSec: number
  /**
   * Anchor scrubber maps the clip's pixel width onto [0, sourceMaxSec].
   * The clip's visual width is independent of source duration, so this is
   * effectively a slider with a configurable range.
   */
  sourceMaxSec: number
  onAnchorChange: (sec: number) => void
  /** Wired only for clip B; clip A stays anchored at x=0. */
  onClipPointerDown?: (e: ReactPointerEvent<HTMLDivElement>) => void
  draggable?: boolean
  disabled?: boolean
  className?: string
}

export function ClipBlock(props: ClipBlockProps) {
  const {
    label,
    source,
    x,
    width,
    anchorSec,
    sourceMaxSec,
    onAnchorChange,
    onClipPointerDown,
    draggable = false,
    disabled = false,
    className,
  } = props

  const clampedAnchor = Math.max(0, Math.min(sourceMaxSec, anchorSec))
  const anchorPx = (clampedAnchor / sourceMaxSec) * width

  const trackRef = useRef<HTMLDivElement>(null)

  const handleAnchorPointerDown = useCallback(
    (e: ReactPointerEvent<HTMLDivElement>) => {
      if (disabled) return
      // Stop bubbling so this doesn't also start a clip drag.
      e.stopPropagation()
      const target = e.currentTarget
      target.setPointerCapture(e.pointerId)

      const updateFromClientX = (clientX: number) => {
        const trackEl = trackRef.current
        if (!trackEl) return
        const rect = trackEl.getBoundingClientRect()
        const local = Math.max(0, Math.min(rect.width, clientX - rect.left))
        const sec = (local / rect.width) * sourceMaxSec
        onAnchorChange(sec)
      }

      updateFromClientX(e.clientX)

      const onMove = (ev: PointerEvent) => updateFromClientX(ev.clientX)
      const onUp = (ev: PointerEvent) => {
        target.releasePointerCapture(ev.pointerId)
        target.removeEventListener('pointermove', onMove)
        target.removeEventListener('pointerup', onUp)
        target.removeEventListener('pointercancel', onUp)
      }
      target.addEventListener('pointermove', onMove)
      target.addEventListener('pointerup', onUp)
      target.addEventListener('pointercancel', onUp)
    },
    [disabled, onAnchorChange, sourceMaxSec],
  )

  return (
    <div
      className={`clip-block${draggable ? ' clip-draggable' : ''}${
        disabled ? ' clip-disabled' : ''
      }${className ? ` ${className}` : ''}`}
      style={{ left: `${x}px`, width: `${width}px` }}
      onPointerDown={draggable ? onClipPointerDown : undefined}
      role="group"
      aria-label={`${label}: ${source}`}
    >
      <div className="clip-header">
        <span className="clip-label">{label}</span>
        <span className="clip-source">{source}</span>
      </div>
      <div ref={trackRef} className="clip-anchor-track">
        <div
          className="clip-anchor-marker"
          style={{ left: `${anchorPx}px` }}
          onPointerDown={handleAnchorPointerDown}
          role="slider"
          aria-label={`${label} anchor`}
          aria-valuemin={0}
          aria-valuemax={sourceMaxSec}
          aria-valuenow={Number(clampedAnchor.toFixed(2))}
        >
          <div className="clip-anchor-stem" />
          <div className="clip-anchor-handle" />
        </div>
      </div>
      <div className="clip-anchor-value">anchor {clampedAnchor.toFixed(2)}s</div>
    </div>
  )
}
