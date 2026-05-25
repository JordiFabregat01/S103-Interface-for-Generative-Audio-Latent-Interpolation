const API_BASE = (
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000"
).replace(/\/$/, "");

export type SoundPoint = {
  id: number;
  name: string;
  filename: string;
  x: number;
  y: number;
};

export const getSounds = (): Promise<SoundPoint[]> =>
  fetch(`${API_BASE}/sounds`).then((res) => {
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  });

export const getSoundUrl = (filename: string): string =>
  `${API_BASE}/sounds/${encodeURIComponent(filename)}`;

export type SoundHit = SoundPoint & { score: number };

export const searchSounds = (q: string, k = 8): Promise<SoundHit[]> =>
  fetch(`${API_BASE}/sounds/search?q=${encodeURIComponent(q)}&k=${k}`).then((res) => {
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  });

export const findSimilarSounds = (filename: string, k = 8): Promise<SoundHit[]> =>
  fetch(`${API_BASE}/sounds/${encodeURIComponent(filename)}/similar?k=${k}`).then((res) => {
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  });

export type InterpolationRequest = {
  audio1: string;
  audio2: string;
  distance_sec?: number;
  duration_sec?: number;
  nfe?: number;
  context_mode?: "auto" | "static_first" | "static_at_anchor" | "dynamic";
};

export type ClipSegment = {
  type: "clip";
  filename: string;
  duration: number;
};

export type SilenceSegment = {
  type: "silence";
  duration: number;
};

export type InterpolationSegment = {
  type: "interpolation";
  audio1: string;
  audio2: string;
  distance_sec?: number;
  duration_sec?: number;
  nfe?: number;
  context_mode?: "auto" | "static_first" | "static_at_anchor" | "dynamic";
};

export type Segment = ClipSegment | SilenceSegment | InterpolationSegment;

/**
 * Render a full timeline composition (N clips, N-1 gaps) to a single WAV.
 * Returns an object URL for the generated WAV. Caller must revoke it when done.
 */
export const render = async (segments: Segment[]): Promise<string> => {
  const res = await fetch(`${API_BASE}/render`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ segments }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? `HTTP ${res.status}`);
  }
  const blob = await res.blob();
  return URL.createObjectURL(blob);
};

/** Returns an object URL for the generated WAV. Caller must revoke it when done. */
export const interpolate = async (req: InterpolationRequest): Promise<string> => {
  const body: Record<string, unknown> = {
    nfe: 8,
    context_mode: "auto",
    ...req,
  };
  const res = await fetch(`${API_BASE}/interpolate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? `HTTP ${res.status}`);
  }
  const blob = await res.blob();
  return URL.createObjectURL(blob);
};
