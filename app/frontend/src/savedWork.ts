export type TimelineClipSnapshot = { // shape of one clip in the timeline
  id: number;
  name: string;
  filename: string;
  start: number;
  duration: number;
};

export type SavedWork = { // shape of a saved work
  id: string;
  name: string;
  savedAt: string;
  timelineClips: TimelineClipSnapshot[];
  interpolatedGaps: string[];
  quality: number;
  durationSec?: number;
};

type StoredWork = SavedWork & { wav: Blob }; 

const DB_NAME = "gali-saved-work";
const DB_VERSION = 1;
const STORE = "works";
const MAX_ENTRIES = 30;

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE)) {
        db.createObjectStore(STORE, { keyPath: "id" });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("Failed to open saved work database"));
  });
}

function runTransaction<T>(
  mode: IDBTransactionMode,
  fn: (store: IDBObjectStore) => IDBRequest<T>,
): Promise<T> {
  return openDb().then(
    (db) =>
      new Promise((resolve, reject) => {
        const tx = db.transaction(STORE, mode);
        const store = tx.objectStore(STORE);
        const request = fn(store);
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error ?? new Error("Saved work database request failed"));
        tx.oncomplete = () => db.close();
        tx.onerror = () => reject(tx.error ?? new Error("Saved work database transaction failed"));
      }),
  );
}

export async function listSavedWorks(): Promise<SavedWork[]> {
  const rows = await runTransaction<StoredWork[]>("readonly", (store) => store.getAll());
  return rows
    .map(({ wav: _wav, ...meta }) => meta)
    .sort((a, b) => b.savedAt.localeCompare(a.savedAt));
}

export async function getSavedWork(id: string): Promise<{ meta: SavedWork; wav: Blob } | null> {
  const row = await runTransaction<StoredWork | undefined>("readonly", (store) => store.get(id));
  if (!row) return null;
  const { wav, ...meta } = row;
  return { meta, wav };
}

export async function saveWork(
  input: Omit<SavedWork, "id" | "savedAt">,
  wav: Blob,
): Promise<SavedWork> {
  const entry: StoredWork = {
    ...input,
    id: crypto.randomUUID(),
    savedAt: new Date().toISOString(),
    wav,
  };

  await runTransaction("readwrite", (store) => store.put(entry));

  const all = await listSavedWorks();
  if (all.length > MAX_ENTRIES) {
    const toRemove = all.slice(MAX_ENTRIES);
    await Promise.all(toRemove.map((item) => deleteSavedWork(item.id)));
  }

  const { wav: _wav, ...meta } = entry;
  return meta;
}

export async function deleteSavedWork(id: string): Promise<void> {
  await runTransaction("readwrite", (store) => store.delete(id));
}

export function defaultSavedWorkName(clips: TimelineClipSnapshot[]): string {
  const sorted = [...clips].sort((a, b) => a.start - b.start);
  if (sorted.length === 0) return "Untitled interpolation";
  const label = sorted.map((clip) => clip.name).join(" → ");
  const date = new Date().toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
  return `${label} · ${date}`;
}

export function timelineEndSec(clips: TimelineClipSnapshot[]): number {
  return clips.reduce((max, clip) => Math.max(max, clip.start + clip.duration), 0);
}
