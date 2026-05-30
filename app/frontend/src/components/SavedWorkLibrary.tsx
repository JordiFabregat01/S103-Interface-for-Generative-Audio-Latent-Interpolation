import { useEffect, useRef, useState } from "react";
import {
  deleteSavedWork,
  getSavedWork,
  listSavedWorks,
  timelineEndSec,
  type SavedWork,
} from "../savedWork";

type Props = {
  open: boolean;
  onClose: () => void;
  onLoad: (work: SavedWork, wavUrl: string) => void;
};

function fmtDate(iso: string) {
  return new Date(iso).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function MiniTimeline({ work }: { work: SavedWork }) {
  const end = Math.max(timelineEndSec(work.timelineClips), 0.01);
  const sorted = [...work.timelineClips].sort((a, b) => a.start - b.start);

  return (
    <div className="saved-work-mini-timeline" aria-hidden="true">
      {sorted.map((clip) => (
        <div
          key={clip.id}
          className="saved-work-mini-clip"
          style={{
            left: `${(clip.start / end) * 100}%`,
            width: `${Math.max((clip.duration / end) * 100, 2)}%`,
          }}
          title={clip.name}
        />
      ))}
    </div>
  );
}

export default function SavedWorkLibrary({ open, onClose, onLoad }: Props) {
  const [items, setItems] = useState<SavedWork[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);
  const previewUrls = useRef<Map<string, string>>(new Map());

  const refresh = async () => {
    setLoading(true);
    setError(null);
    try {
      setItems(await listSavedWorks());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load saved work");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!open) return;
    refresh();
  }, [open]);

  useEffect(() => {
    return () => {
      for (const url of previewUrls.current.values()) URL.revokeObjectURL(url);
      previewUrls.current.clear();
    };
  }, []);

  const handleLoad = async (id: string) => {
    setBusyId(id);
    setError(null);
    try {
      const row = await getSavedWork(id);
      if (!row) {
        setError("This saved work could not be found.");
        return;
      }
      let url = previewUrls.current.get(id);
      if (!url) {
        url = URL.createObjectURL(row.wav);
        previewUrls.current.set(id, url);
      }
      onLoad(row.meta, url);
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load saved work");
    } finally {
      setBusyId(null);
    }
  };

  const handleDownload = async (work: SavedWork) => {
    setBusyId(work.id);
    setError(null);
    try {
      const row = await getSavedWork(work.id);
      if (!row) {
        setError("This saved work could not be found.");
        return;
      }
      const url = URL.createObjectURL(row.wav);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `${work.name.replace(/[^\w\- ]+/g, "").trim() || "interpolation"}.wav`;
      anchor.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to download WAV");
    } finally {
      setBusyId(null);
    }
  };

  const handleDelete = async (id: string) => {
    setBusyId(id);
    setError(null);
    try {
      await deleteSavedWork(id);
      const url = previewUrls.current.get(id);
      if (url) {
        URL.revokeObjectURL(url);
        previewUrls.current.delete(id);
      }
      setItems((prev) => prev.filter((item) => item.id !== id));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete saved work");
    } finally {
      setBusyId(null);
    }
  };

  if (!open) return null;

  return (
    <>
      <div className="info-modal-backdrop" onClick={onClose} />
      <div className="info-modal saved-work-modal" role="dialog" aria-modal="true" aria-label="Saved work">
        <div className="info-modal-header">
          <h3>Saved Work</h3>
          <button className="close-preview-btn" onClick={onClose}>✕</button>
        </div>
        <div className="info-modal-body saved-work-body">
          <p className="saved-work-intro">
            Saved interpolations keep both the rendered WAV and the timeline layout used to create it.
          </p>

          {error && <p className="interp-error">{error}</p>}
          {loading && <p className="library-hint">Loading saved work…</p>}

          {!loading && items.length === 0 && (
            <p className="saved-work-empty">
              Nothing saved yet. After you interpolate, use <strong>Save to library</strong> in the timeline header.
            </p>
          )}

          {!loading && items.length > 0 && (
            <ul className="saved-work-list">
              {items.map((work) => {
                const clipCount = work.timelineClips.length;
                const gapCount = work.interpolatedGaps.length;
                const duration = work.durationSec ?? timelineEndSec(work.timelineClips);
                const clipLabel = work.timelineClips
                  .slice()
                  .sort((a, b) => a.start - b.start)
                  .map((clip) => clip.name)
                  .join(" · ");

                return (
                  <li key={work.id} className="saved-work-card">
                    <div className="saved-work-card-main">
                      <div className="saved-work-card-top">
                        <strong className="saved-work-name">{work.name}</strong>
                        <span className="saved-work-date">{fmtDate(work.savedAt)}</span>
                      </div>
                      <MiniTimeline work={work} />
                      <p className="saved-work-meta">
                        {clipCount} clip{clipCount === 1 ? "" : "s"}
                        {gapCount > 0 && ` · ${gapCount} interpolated gap${gapCount === 1 ? "" : "s"}`}
                        {" · "}
                        quality {work.quality}
                        {" · "}
                        {duration.toFixed(1)}s
                      </p>
                      {clipLabel && <p className="saved-work-clips">{clipLabel}</p>}
                    </div>
                    <div className="saved-work-actions">
                      <button
                        className="saved-work-btn primary"
                        onClick={() => handleLoad(work.id)}
                        disabled={busyId === work.id}
                      >
                        Load
                      </button>
                      <button
                        className="saved-work-btn"
                        onClick={() => handleDownload(work)}
                        disabled={busyId === work.id}
                      >
                        WAV
                      </button>
                      <button
                        className="saved-work-btn danger"
                        onClick={() => handleDelete(work.id)}
                        disabled={busyId === work.id}
                        title="Delete"
                      >
                        ✕
                      </button>
                    </div>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      </div>
    </>
  );
}
