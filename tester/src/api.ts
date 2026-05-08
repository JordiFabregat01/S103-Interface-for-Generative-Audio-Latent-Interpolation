import type { InterpolationRequest } from './types'

export interface InterpolationResponse {
  audioUrl: string
  contentType: string
  byteLength: number
  elapsedMs: number
}

export class InterpolationError extends Error {
  status: number
  detail: string
  constructor(status: number, detail: string) {
    super(detail || `HTTP ${status}`)
    this.status = status
    this.detail = detail
    this.name = 'InterpolationError'
  }
}

/**
 * POST /api/interpolate. Vite forwards this to http://localhost:8000/interpolate
 * (see vite.config.ts). On 2xx returns a Blob URL the caller is responsible for
 * revoking via revokeAudioUrl when it's swapped out.
 */
export async function postInterpolate(
  req: InterpolationRequest,
  signal?: AbortSignal,
): Promise<InterpolationResponse> {
  const startedAt = performance.now()
  const res = await fetch('/api/interpolate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
    signal,
  })

  if (!res.ok) {
    const detail = await extractErrorDetail(res)
    throw new InterpolationError(res.status, detail)
  }

  const blob = await res.blob()
  const audioUrl = URL.createObjectURL(blob)
  return {
    audioUrl,
    contentType: blob.type || 'audio/wav',
    byteLength: blob.size,
    elapsedMs: performance.now() - startedAt,
  }
}

export function revokeAudioUrl(url: string | null): void {
  if (url) URL.revokeObjectURL(url)
}

async function extractErrorDetail(res: Response): Promise<string> {
  try {
    const text = await res.text()
    if (!text) return `HTTP ${res.status}`
    try {
      const parsed = JSON.parse(text) as unknown
      return formatFastApiDetail(parsed) ?? text
    } catch {
      return text
    }
  } catch {
    return `HTTP ${res.status}`
  }
}

// FastAPI 422 returns `{ detail: [{ loc, msg, type }, ...] }` for validation
// errors and `{ detail: "..." }` for HTTPException. Normalize both to a string
// so the UI can show a single status line.
function formatFastApiDetail(parsed: unknown): string | null {
  if (
    typeof parsed === 'object' &&
    parsed !== null &&
    'detail' in parsed
  ) {
    const detail = (parsed as { detail: unknown }).detail
    if (typeof detail === 'string') return detail
    if (Array.isArray(detail)) {
      return detail
        .map((item) => {
          if (typeof item === 'object' && item !== null && 'msg' in item) {
            const loc = 'loc' in item && Array.isArray((item as { loc: unknown[] }).loc)
              ? (item as { loc: unknown[] }).loc.join('.')
              : ''
            const msg = String((item as { msg: unknown }).msg)
            return loc ? `${loc}: ${msg}` : msg
          }
          return JSON.stringify(item)
        })
        .join('; ')
    }
  }
  return null
}
