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

export type JobProgress = { done: number; total: number };

export type JobSnapshot =
  | { status: "queued"; progress?: JobProgress }
  | { status: "running"; progress?: JobProgress }
  | { status: "done"; result: { url: string; bytes: number } }
  | { status: "error"; error?: string }
  | { status: "cancelled"; error?: string };

export type RenderOptions = {
  signal?: AbortSignal;
  onJobId?: (jobId: string) => void;
  onProgress?: (snapshot: JobSnapshot) => void;
  pollIntervalMs?: number;
};

const DEFAULT_POLL_MS = 500;

const parseDetail = async (res: Response): Promise<string> => {
  const err = await res.json().catch(() => ({}));
  return (err as { detail?: string }).detail ?? `HTTP ${res.status}`;
};

const submitJob = async (path: string, body: unknown, signal?: AbortSignal): Promise<string> => {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) throw new Error(await parseDetail(res));
  const json = (await res.json()) as { job_id?: string };
  if (!json.job_id) throw new Error("server did not return a job_id");
  return json.job_id;
};

const pollJob = async (jobId: string, signal?: AbortSignal): Promise<JobSnapshot> => {
  const res = await fetch(`${API_BASE}/jobs/${jobId}`, { signal });
  if (!res.ok) throw new Error(await parseDetail(res));
  return (await res.json()) as JobSnapshot;
};

const wait = (ms: number, signal?: AbortSignal): Promise<void> =>
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

const waitForJob = async (
  jobId: string,
  { signal, onProgress, pollIntervalMs = DEFAULT_POLL_MS }: RenderOptions = {},
): Promise<JobSnapshot> => {
  // eslint-disable-next-line no-constant-condition
  while (true) {
    const snapshot = await pollJob(jobId, signal);
    onProgress?.(snapshot);
    if (
      snapshot.status === "done" ||
      snapshot.status === "error" ||
      snapshot.status === "cancelled"
    ) {
      return snapshot;
    }
    await wait(pollIntervalMs, signal);
  }
};

export const cancelRender = async (jobId: string): Promise<void> => {
  const res = await fetch(`${API_BASE}/jobs/${jobId}/cancel`, { method: "POST" });
  if (!res.ok) throw new Error(await parseDetail(res));
};

/**
 * Submit a timeline render to the backend, poll until it finishes, then
 * fetch the rendered WAV. Returns an object URL for the WAV; caller must
 * revoke it when done.
 */
export const render = async (
  segments: Segment[],
  options: RenderOptions = {},
): Promise<string> => {
  const jobId = await submitJob("/render", { segments }, options.signal);
  options.onJobId?.(jobId);
  const snapshot = await waitForJob(jobId, options);
  if (snapshot.status === "error" || snapshot.status === "cancelled") {
    throw new Error(snapshot.error ?? `render ${snapshot.status}`);
  }
  if (snapshot.status !== "done") {
    throw new Error(`render did not finish (status=${snapshot.status})`);
  }
  const res = await fetch(`${API_BASE}/jobs/${jobId}/result.wav`, {
    signal: options.signal,
  });
  if (!res.ok) throw new Error(await parseDetail(res));
  const blob = await res.blob();
  return URL.createObjectURL(blob);
};

/** Returns an object URL for the generated WAV. Caller must revoke it when done. */
export const interpolate = async (
  req: InterpolationRequest,
  options: RenderOptions = {},
): Promise<string> => {
  const body: Record<string, unknown> = {
    nfe: 8,
    context_mode: "auto",
    ...req,
  };
  const jobId = await submitJob("/interpolate", body, options.signal);
  options.onJobId?.(jobId);
  const snapshot = await waitForJob(jobId, options);
  if (snapshot.status === "error" || snapshot.status === "cancelled") {
    throw new Error(snapshot.error ?? `interpolate ${snapshot.status}`);
  }
  if (snapshot.status !== "done") {
    throw new Error(`interpolate did not finish (status=${snapshot.status})`);
  }
  const res = await fetch(`${API_BASE}/jobs/${jobId}/result.wav`, {
    signal: options.signal,
  });
  if (!res.ok) throw new Error(await parseDetail(res));
  const blob = await res.blob();
  return URL.createObjectURL(blob);
};
