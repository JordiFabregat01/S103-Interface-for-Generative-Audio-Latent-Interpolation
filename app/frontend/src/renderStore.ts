// Tiny IndexedDB-backed store for rendered audio so the Renders tab survives a
// page refresh. We persist the actual WAV Blob plus the timeline metadata, and
// rebuild fresh object URLs on load. A cap keeps storage from growing forever.

import type { Kind } from "./api";

export type StoredClip = {
  id: number;
  name: string;
  filename: string;
  kind: Kind;
  start: number;
  duration: number;
};

export type RenderRecord = {
  id: number;
  name: string;
  clips: StoredClip[];
  gaps: string[];
  blob: Blob;
  /** Wall-clock render time in milliseconds (optional for older records). */
  renderMs?: number;
};

const DB_NAME = "gali-renders";
const STORE = "renders";
const MAX_RENDERS = 5;

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) {
        db.createObjectStore(STORE, { keyPath: "id" });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function tx(db: IDBDatabase, mode: IDBTransactionMode): IDBObjectStore {
  return db.transaction(STORE, mode).objectStore(STORE);
}

/** All stored renders, newest first (id is a timestamp). */
export async function getAllRenders(): Promise<RenderRecord[]> {
  const db = await openDb();
  try {
    const records = await new Promise<RenderRecord[]>((resolve, reject) => {
      const req = tx(db, "readonly").getAll();
      req.onsuccess = () => resolve(req.result as RenderRecord[]);
      req.onerror = () => reject(req.error);
    });
    return records.sort((a, b) => b.id - a.id);
  } finally {
    db.close();
  }
}

/** Save a render, then prune to the newest MAX_RENDERS. */
export async function putRender(record: RenderRecord): Promise<void> {
  const db = await openDb();
  try {
    await new Promise<void>((resolve, reject) => {
      const store = tx(db, "readwrite");
      store.put(record);
      store.transaction.oncomplete = () => resolve();
      store.transaction.onerror = () => reject(store.transaction.error);
    });
    await prune(db);
  } finally {
    db.close();
  }
}

export async function deleteRender(id: number): Promise<void> {
  const db = await openDb();
  try {
    await new Promise<void>((resolve, reject) => {
      const store = tx(db, "readwrite");
      store.delete(id);
      store.transaction.oncomplete = () => resolve();
      store.transaction.onerror = () => reject(store.transaction.error);
    });
  } finally {
    db.close();
  }
}

function prune(db: IDBDatabase): Promise<void> {
  return new Promise((resolve, reject) => {
    const store = tx(db, "readwrite");
    const keysReq = store.getAllKeys();
    keysReq.onsuccess = () => {
      const keys = (keysReq.result as number[]).sort((a, b) => b - a);
      for (const key of keys.slice(MAX_RENDERS)) store.delete(key);
    };
    store.transaction.oncomplete = () => resolve();
    store.transaction.onerror = () => reject(store.transaction.error);
  });
}

export { MAX_RENDERS };
