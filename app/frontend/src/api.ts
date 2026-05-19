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

export type SoundHit = SoundPoint & { score: number };

export const searchSounds = (q: string, k = 8): Promise<SoundHit[]> =>
  fetch(
    `${API_BASE}/sounds/search?q=${encodeURIComponent(q)}&k=${k}`
  ).then((res) => {
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  });

export const findSimilarSounds = (filename: string, k = 8): Promise<SoundHit[]> =>
  fetch(
    `${API_BASE}/sounds/${encodeURIComponent(filename)}/similar?k=${k}`
  ).then((res) => {
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

const buildInterpolationBody = (req: InterpolationRequest): Record<string, unknown> => ({
  distance_sec: 0.0,
  duration_sec: 3.0,
  nfe: 8,
  context_mode: "auto",
  ...req,
});

/** Returns an object URL for the generated WAV. Caller must revoke it when done. */
export const interpolate = async (req: InterpolationRequest): Promise<string> => {
  const res = await fetch(`${API_BASE}/interpolate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(buildInterpolationBody(req)),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? `HTTP ${res.status}`);
  }
  const blob = await res.blob();
  return URL.createObjectURL(blob);
};

// ---------------------------------------------------------------------------
// Async render jobs
// ---------------------------------------------------------------------------

export type JobStatus = "queued" | "running" | "done" | "error" | "cancelled";

export type JobProgress = { done: number; total: number };

export type JobSnapshot =
  | { status: "queued" | "running"; progress: JobProgress }
  | { status: "done"; result: { url: string; bytes: number } }
  | { status: "error" | "cancelled"; error?: string };

export const startRender = async (req: InterpolationRequest): Promise<string> => {
  const res = await fetch(`${API_BASE}/render`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(buildInterpolationBody(req)),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? `HTTP ${res.status}`);
  }
  const data = (await res.json()) as { job_id: string };
  return data.job_id;
};

export const getJobStatus = async (jobId: string): Promise<JobSnapshot> => {
  const res = await fetch(`${API_BASE}/jobs/${encodeURIComponent(jobId)}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? `HTTP ${res.status}`);
  }
  return (await res.json()) as JobSnapshot;
};

export const getJobResultUrl = (jobId: string): string =>
  `${API_BASE}/jobs/${encodeURIComponent(jobId)}/result.wav`;

export const cancelJob = async (jobId: string): Promise<void> => {
  const res = await fetch(`${API_BASE}/jobs/${encodeURIComponent(jobId)}/cancel`, {
    method: "POST",
  });
  if (!res.ok && res.status !== 404) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? `HTTP ${res.status}`);
  }
};

export type RunRenderJobOptions = {
  onProgress?: (snapshot: JobSnapshot, jobId: string) => void;
  signal?: AbortSignal;
  pollIntervalMs?: number;
};

export type RenderJobResult = {
  jobId: string;
  url: string;
  bytes: number;
};

const sleep = (ms: number, signal?: AbortSignal): Promise<void> =>
  new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException("aborted", "AbortError"));
      return;
    }
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(timer);
      reject(new DOMException("aborted", "AbortError"));
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  });

/**
 * High-level helper: start a render, poll until it terminates, and return the
 * playable result URL. Cancellation via `signal.abort()` will both stop the
 * poll loop and ask the backend to drop the job.
 */
export const runRenderJob = async (
  req: InterpolationRequest,
  { onProgress, signal, pollIntervalMs = 400 }: RunRenderJobOptions = {}
): Promise<RenderJobResult> => {
  const jobId = await startRender(req);

  const cancelOnAbort = () => {
    void cancelJob(jobId).catch(() => undefined);
  };
  if (signal?.aborted) {
    cancelOnAbort();
    throw new DOMException("aborted", "AbortError");
  }
  signal?.addEventListener("abort", cancelOnAbort, { once: true });

  try {
    while (true) {
      const snapshot = await getJobStatus(jobId);
      onProgress?.(snapshot, jobId);

      if (snapshot.status === "done") {
        return {
          jobId,
          url: getJobResultUrl(jobId),
          bytes: snapshot.result.bytes,
        };
      }
      if (snapshot.status === "error") {
        throw new Error(snapshot.error ?? "render failed");
      }
      if (snapshot.status === "cancelled") {
        throw new DOMException(
          snapshot.error ?? "render cancelled",
          "AbortError"
        );
      }

      await sleep(pollIntervalMs, signal);
    }
  } finally {
    signal?.removeEventListener("abort", cancelOnAbort);
  }
};
