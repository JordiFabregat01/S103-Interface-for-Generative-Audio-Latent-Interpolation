import type { ReactNode } from 'react'
import {
  AUDIO_VALUES,
  CONTEXT_MODES,
  DECODE_METHODS,
  type AudioValue,
  type ContextMode,
  type DecodeMethod,
} from '../types'

export interface ParamPatch {
  audio1?: AudioValue
  audio2?: AudioValue
  duration_sec?: number | null
  stay_time_sec?: number
  stickyness?: number
  nfe?: number
  context_mode?: ContextMode
  decode_method?: DecodeMethod
}

export interface ParamPanelProps {
  audio1: AudioValue
  audio2: AudioValue
  /** User-provided value used only when distance_sec === 0. */
  duration_sec: number | null
  stay_time_sec: number
  stickyness: number
  nfe: number
  context_mode: ContextMode
  decode_method: DecodeMethod
  /** Read-only here. Drives whether duration_sec is editable. */
  distance_sec: number
  onChange: (patch: ParamPatch) => void
  disabled?: boolean
}

export function ParamPanel(props: ParamPanelProps) {
  const {
    audio1,
    audio2,
    duration_sec,
    stay_time_sec,
    stickyness,
    nfe,
    context_mode,
    decode_method,
    distance_sec,
    onChange,
    disabled = false,
  } = props

  const adjacent = distance_sec === 0
  // When non-adjacent the backend ignores duration_sec and uses |distance_sec|.
  // Show that derived value (read-only) so the user knows what's effective.
  const displayedDuration = adjacent ? duration_sec ?? '' : Math.abs(distance_sec)

  return (
    <section className="panel param-panel" aria-label="Engine parameters">
      <div className="panel-header">Parameters</div>

      <div className="param-grid">
        <Field label="audio1">
          <select
            value={audio1}
            disabled={disabled}
            onChange={(e) => onChange({ audio1: e.target.value as AudioValue })}
          >
            {AUDIO_VALUES.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </Field>

        <Field label="audio2">
          <select
            value={audio2}
            disabled={disabled}
            onChange={(e) => onChange({ audio2: e.target.value as AudioValue })}
          >
            {AUDIO_VALUES.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </Field>

        <Field
          label="duration_sec"
          hint={
            adjacent
              ? 'Required (distance_sec is 0)'
              : `Auto: |distance_sec| = ${Math.abs(distance_sec).toFixed(2)}s`
          }
        >
          <input
            type="number"
            step="0.05"
            min="0"
            inputMode="decimal"
            disabled={disabled || !adjacent}
            value={displayedDuration}
            onChange={(e) => {
              const raw = e.target.value
              if (raw === '') {
                onChange({ duration_sec: null })
                return
              }
              const num = Number(raw)
              if (!Number.isNaN(num)) onChange({ duration_sec: num })
            }}
          />
        </Field>

        <Field label="stay_time_sec">
          <input
            type="number"
            step="0.05"
            min="0"
            inputMode="decimal"
            disabled={disabled}
            value={stay_time_sec}
            onChange={(e) => {
              const num = Number(e.target.value)
              if (!Number.isNaN(num)) onChange({ stay_time_sec: num })
            }}
          />
        </Field>

        <Field label="stickyness">
          <input
            type="number"
            step="0.1"
            min="0.01"
            inputMode="decimal"
            disabled={disabled}
            value={stickyness}
            onChange={(e) => {
              const num = Number(e.target.value)
              if (!Number.isNaN(num)) onChange({ stickyness: num })
            }}
          />
        </Field>

        <Field label="nfe" hint="ODE steps">
          <input
            type="number"
            step="1"
            min="1"
            value={nfe}
            disabled={disabled}
            onChange={(e) => {
              const num = Number(e.target.value)
              if (!Number.isNaN(num) && Number.isInteger(num)) {
                onChange({ nfe: num })
              }
            }}
          />
        </Field>

        <Field label="context_mode">
          <select
            value={context_mode}
            disabled={disabled}
            onChange={(e) =>
              onChange({ context_mode: e.target.value as ContextMode })
            }
          >
            {CONTEXT_MODES.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </Field>

        <Field label="decode_method">
          <select
            value={decode_method}
            disabled={disabled}
            onChange={(e) =>
              onChange({ decode_method: e.target.value as DecodeMethod })
            }
          >
            {DECODE_METHODS.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </Field>
      </div>
    </section>
  )
}

function Field(props: {
  label: string
  hint?: string
  children: ReactNode
}) {
  return (
    <label className="param-field">
      <span className="param-label">{props.label}</span>
      {props.children}
      {props.hint ? <span className="param-hint">{props.hint}</span> : null}
    </label>
  )
}
