const API_BASE = "http://localhost:8000";

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

export type InterpolationRequest = {
  audio1: string;
  audio2: string;
  distance_sec?: number;
  duration_sec?: number;
  nfe?: number;
  context_mode?: "auto" | "static_first" | "static_at_anchor" | "dynamic";
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
