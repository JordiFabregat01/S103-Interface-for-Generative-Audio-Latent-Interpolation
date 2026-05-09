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
