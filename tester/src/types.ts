// Mirror of app/backend/inference/models.py InterpolationElement.
// Keep in lockstep with the Pydantic model so the request schema stays valid.

export const AUDIO_VALUES = ['camp_fire', 'keyboard'] as const
export type AudioValue = (typeof AUDIO_VALUES)[number]

export const CONTEXT_MODES = [
  'auto',
  'static_first',
  'static_at_anchor',
  'dynamic',
] as const
export type ContextMode = (typeof CONTEXT_MODES)[number]

export const DECODE_METHODS = ['ola_smooth', 'concat'] as const
export type DecodeMethod = (typeof DECODE_METHODS)[number]

export interface InterpolationRequest {
  audio1: AudioValue
  audio2: AudioValue
  distance_sec: number
  /** Required when distance_sec === 0; ignored otherwise. */
  duration_sec: number | null
  a_anchor_sec: number
  b_anchor_sec: number
  stay_time_sec: number
  stickyness: number
  nfe: number
  context_mode: ContextMode
  decode_method: DecodeMethod
}

export type RenderStatus = 'idle' | 'rendering' | 'error' | 'done'

/**
 * Mirrors `request_from_clip_geometry` in app/backend/inference/interpolation.py:
 *   distance_sec < 0 -> "dynamic"
 *   distance_sec > 0 -> "static_at_anchor"
 *   distance_sec === 0 -> "static_at_anchor"
 * Used purely for the live "resolved mode" readout in the UI.
 */
export function resolveContextMode(
  override: ContextMode,
  distance_sec: number,
): Exclude<ContextMode, 'auto'> {
  if (override !== 'auto') return override
  return distance_sec < 0 ? 'dynamic' : 'static_at_anchor'
}

/**
 * Effective interpolation duration the backend will use for the bridge.
 * Mirrors the geometry helper: |distance| for non-zero distance, otherwise the
 * caller-provided `duration_sec` (which the validator requires to be > 0).
 */
export function resolveDurationSec(
  distance_sec: number,
  duration_sec: number | null,
): number | null {
  if (distance_sec !== 0) return Math.abs(distance_sec)
  if (duration_sec !== null && duration_sec > 0) return duration_sec
  return null
}
