import { useEffect, useMemo, useRef, useState } from "react";
import "../App.css";
import { getSounds, getSoundUrl, interpolate, type SoundPoint } from "../api";
import { useAudioPlayer } from "../hooks/useAudioPlayer";

type TimelineClip = {
  id: number;
  name: string;
  filename: string;
  start: number;
  duration: number;
};

const PX_PER_SEC = 80;
const CLIP_WIDTH_PX = 240;
const TIMELINE_BUFFER_PX = 400;

const VALID_AUDIO_ELEMENTS = new Set(["camp_fire", "keyboard"]);

function fmt(sec: number) {
  if (!isFinite(sec)) return "0:00";
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function filenameToAudioElement(filename: string) {
  return filename.replace(/\.[^.]+$/, "").toLowerCase().replace(/\s+/g, "_");
}

export default function WorkspaceView() {
  const [sounds, setSounds] = useState<SoundPoint[]>([]);
  const [timelineClips, setTimelineClips] = useState<TimelineClip[]>([]);
  const [draggingId, setDraggingId] = useState<number | null>(null);
  const [selectedSound, setSelectedSound] = useState<SoundPoint | null>(null);
  const [explorerSelected, setExplorerSelected] = useState<SoundPoint | null>(null);
  const previewPlayer = useAudioPlayer();
  const interpPlayer = useAudioPlayer();
  const explorerPlayer = useAudioPlayer();
  const dragSoundRef = useRef<SoundPoint | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const [containerWidth, setContainerWidth] = useState(0);
  const [dragExtentPx, setDragExtentPx] = useState(0);
  const [dragStartWidth, setDragStartWidth] = useState(0);
  const autoScrollRaf = useRef<number | null>(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => setContainerWidth(entry.contentRect.width));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const stopAutoScroll = () => {
    if (autoScrollRaf.current !== null) {
      cancelAnimationFrame(autoScrollRaf.current);
      autoScrollRaf.current = null;
    }
  };

  const resetDragState = () => {
    stopAutoScroll();
    setDragExtentPx(0);
  };

  const handleTimelineDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    const scroll = scrollRef.current;
    if (!scroll) return;

    const EDGE_ZONE = 80;
    const SCROLL_SPEED = 6;
    const rect = scroll.getBoundingClientRect();

    if (e.clientX > rect.right - EDGE_ZONE) {
      if (autoScrollRaf.current === null) {
        const tick = () => {
          const s = scrollRef.current;
          if (!s) return;
          s.scrollLeft += SCROLL_SPEED;
          setDragExtentPx(s.scrollLeft + s.clientWidth);
          autoScrollRaf.current = requestAnimationFrame(tick);
        };
        autoScrollRaf.current = requestAnimationFrame(tick);
      }
    } else {
      stopAutoScroll();
    }
  };

  const [interpLoading, setInterpLoading] = useState(false);
  const [interpError, setInterpError] = useState<string | null>(null);
  const [interpUrl, setInterpUrl] = useState<string | null>(null);
  const [interpDuration, setInterpDuration] = useState(3.0);
  const interpUrlRef = useRef<string | null>(null);

  useEffect(() => {
    getSounds()
      .then(setSounds)
      .catch((err) => console.error("Error loading sounds:", err));
  }, []);

  const getEmoji = (name: string) => {
    const lower = name.toLowerCase();
    if (lower.includes("rain")) return "🌧️";
    if (lower.includes("bird")) return "🐦";
    if (lower.includes("water")) return "🌊";
    if (lower.includes("wind")) return "💨";
    if (lower.includes("fire")) return "🔥";
    if (lower.includes("thunder")) return "⚡";
    if (lower.includes("keyboard")) return "⌨️";
    if (lower.includes("foot")) return "👣";
    return "🎵";
  };

  const moveClip = (id: number, newStart: number) => {
    setTimelineClips((prev) =>
      prev.map((clip) =>
        clip.id === id ? { ...clip, start: Math.max(0, newStart) } : clip
      )
    );
  };

  const runInterpolation = async () => {
    const sorted = [...timelineClips].sort((a, b) => a.start - b.start);
    const [clipA, clipB] = sorted;
    if (!clipA || !clipB) return;

    const a1 = filenameToAudioElement(clipA.filename);
    const a2 = filenameToAudioElement(clipB.filename);
    const unsupported = [
      !VALID_AUDIO_ELEMENTS.has(a1) && clipA.name,
      !VALID_AUDIO_ELEMENTS.has(a2) && clipB.name,
    ].filter(Boolean);
    if (unsupported.length > 0) {
      setInterpError(
        `Not supported yet: ${unsupported.join(", ")}. Currently only "Camp Fire" and "Keyboard" can be interpolated.`
      );
      return;
    }

    setInterpLoading(true);
    setInterpError(null);
    if (interpUrlRef.current) {
      URL.revokeObjectURL(interpUrlRef.current);
      interpUrlRef.current = null;
    }
    setInterpUrl(null);
    try {
      const url = await interpolate({
        audio1: a1,
        audio2: a2,
        distance_sec: 0.0,
        duration_sec: interpDuration,
      });
      interpUrlRef.current = url;
      setInterpUrl(url);
    } catch (err) {
      setInterpError(err instanceof Error ? err.message : "Interpolation failed");
    } finally {
      setInterpLoading(false);
    }
  };

  const overlaps: number[] = [];
  [...timelineClips].sort((a, b) => a.start - b.start).forEach((clip, index, arr) => {
    const next = arr[index + 1];
    if (next && clip.start < next.start + next.duration && clip.start + clip.duration > next.start) {
      overlaps.push(clip.id, next.id);
    }
  });

  const rightmostPx = timelineClips
    .filter((c) => c.id !== draggingId)
    .reduce((max, c) => Math.max(max, c.start + c.duration), 0);
  const snapUp = (px: number) => Math.ceil(px / PX_PER_SEC) * PX_PER_SEC;
  const clipWidth = rightmostPx > 0 ? snapUp(rightmostPx + TIMELINE_BUFFER_PX) : 0;
  const dragWidth = dragExtentPx > containerWidth ? snapUp(dragExtentPx + TIMELINE_BUFFER_PX) : 0;
  const timelineWidth = Math.max(containerWidth, clipWidth, dragWidth, draggingId ? dragStartWidth : 0);
  const rulerSeconds = Math.ceil(timelineWidth / PX_PER_SEC) + 1;

  const canInterpolate = timelineClips.length >= 2;

  const placedPoints = useMemo(() => sounds.map((p) => ({ ...p, px: p.x * 100, py: p.y * 100 })), [sounds]);

  const deleteClip = (id: number) => {
    setTimelineClips((prev) => prev.filter((c) => c.id !== id));
  };

  return (
    <div className="workspace-page">
      <div className="app-header">
        <h1>Generative Audio Latent Interpolation</h1>
      </div>

      <div className="workspace-layout">
        <div className="library-panel">
          <h2>Sound Library</h2>
          <p className="library-hint">Click to preview · Drag to timeline</p>

          <div className="sound-grid-scroll">
            <div className="sound-grid">
              {sounds.map((sound) => (
                <div
                  key={sound.id}
                  className={`sound-card${selectedSound?.id === sound.id ? " selected" : ""}`}
                  draggable
                  onClick={() => {
                    if (selectedSound?.id === sound.id) {
                      previewPlayer.pause();
                      setSelectedSound(null);
                    } else {
                      interpPlayer.pause();
                      setSelectedSound(sound);
                      previewPlayer.play(getSoundUrl(sound.filename));
                    }
                  }}
                  onDragStart={() => { dragSoundRef.current = sound; }}
                  onDragEnd={() => { dragSoundRef.current = null; resetDragState(); }}
                >
                  <div className="sound-image">{getEmoji(sound.name)}</div>
                  <p>{sound.name}</p>
                </div>
              ))}
            </div>
          </div>

          {selectedSound && (
            <div className="sound-preview-panel">
              <div className="sound-preview-header">
                <span className="preview-name">{getEmoji(selectedSound.name)} {selectedSound.name}</span>
                <button className="close-preview-btn" onClick={() => { previewPlayer.pause(); setSelectedSound(null); }}>✕</button>
              </div>
              <div className="preview-controls">
                <button
                  className="preview-play-btn"
                  onClick={() => { interpPlayer.pause(); previewPlayer.seek(0); previewPlayer.play(getSoundUrl(selectedSound.filename)); }}
                  title="Restart"
                >↺</button>
                <button
                  className="preview-play-btn"
                  onClick={() => previewPlayer.isPlaying ? previewPlayer.pause() : (interpPlayer.pause(), previewPlayer.play(getSoundUrl(selectedSound.filename)))}
                >
                  {previewPlayer.isPlaying ? "⏸" : "▶"}
                </button>
                <input
                  className="audio-scrubber"
                  type="range"
                  min={0}
                  max={previewPlayer.duration || 1}
                  step={0.01}
                  value={previewPlayer.currentTime}
                  onChange={(e) => previewPlayer.seek(Number(e.target.value))}
                />
                <span className="audio-time">{fmt(previewPlayer.currentTime)} / {fmt(previewPlayer.duration)}</span>
              </div>
            </div>
          )}
        </div>

        <div className="drop-panel">
          <h2>Latent Space Exploration</h2>
          <p className="library-hint">Click to preview · Drag to timeline</p>
          <div className="explorer-plot-wrap">
            <div className="explorer-plot">
              {placedPoints.map((point) => (
                <div
                  key={point.id}
                  className={`dot${explorerSelected?.id === point.id ? " selected" : ""}`}
                  style={{ left: `${point.px}%`, top: `${point.py}%` }}
                  draggable
                  onDragStart={() => { dragSoundRef.current = point; }}
                  onDragEnd={() => { dragSoundRef.current = null; resetDragState(); }}
                  onClick={() => {
                    if (explorerSelected?.id === point.id) {
                      if (explorerPlayer.isPlaying) { explorerPlayer.pause(); } else { explorerPlayer.play(getSoundUrl(point.filename)); }
                    } else {
                      explorerPlayer.pause();
                      setExplorerSelected(point);
                      explorerPlayer.play(getSoundUrl(point.filename));
                    }
                  }}
                  title={point.name}
                >
                  <span className="dot-marker">●</span>
                  <span className="dot-label">{point.name}</span>
                </div>
              ))}
            </div>
          </div>
          {explorerSelected && (
            <div className="sound-preview-panel">
              <div className="sound-preview-header">
                <span className="preview-name">{getEmoji(explorerSelected.name)} {explorerSelected.name}</span>
                <button className="close-preview-btn" onClick={() => { explorerPlayer.pause(); setExplorerSelected(null); }}>✕</button>
              </div>
              <div className="preview-controls">
                <button className="preview-play-btn" onClick={() => { explorerPlayer.seek(0); explorerPlayer.play(getSoundUrl(explorerSelected.filename)); }} title="Restart">↺</button>
                <button className="preview-play-btn" onClick={() => explorerPlayer.isPlaying ? explorerPlayer.pause() : explorerPlayer.play(getSoundUrl(explorerSelected.filename))}>
                  {explorerPlayer.isPlaying ? "⏸" : "▶"}
                </button>
                <input className="audio-scrubber" type="range" min={0} max={explorerPlayer.duration || 1} step={0.01} value={explorerPlayer.currentTime} onChange={(e) => explorerPlayer.seek(Number(e.target.value))} />
                <span className="audio-time">{fmt(explorerPlayer.currentTime)} / {fmt(explorerPlayer.duration)}</span>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="timeline-panel">
        <div className="timeline-header">
          <div className="timeline-header-left">
            <h2>Timeline</h2>
            {interpError && <p className="interp-error">{interpError}</p>}
            {interpUrl && !interpLoading && (
              <div className="interp-result">
                <span>Result ready</span>
                <button className="preview-play-btn" onClick={() => { interpPlayer.seek(0); previewPlayer.pause(); interpPlayer.play(interpUrl); }} title="Replay">↺</button>
                <button className="preview-play-btn" onClick={() => { if (interpPlayer.isPlaying) { interpPlayer.pause(); } else { previewPlayer.pause(); interpPlayer.play(interpUrl); } }}>
                  {interpPlayer.isPlaying ? "⏸" : "▶"}
                </button>
                <input
                  className="audio-scrubber"
                  type="range"
                  min={0}
                  max={interpPlayer.duration || 1}
                  step={0.01}
                  value={interpPlayer.currentTime}
                  onChange={(e) => interpPlayer.seek(Number(e.target.value))}
                />
                <span className="audio-time">{fmt(interpPlayer.currentTime)} / {fmt(interpPlayer.duration)}</span>
              </div>
            )}
          </div>
          <div className="timeline-header-right">
            <label className="duration-label">
              Duration: <strong>{interpDuration.toFixed(1)}s</strong>
              <input
                type="range"
                min={1}
                max={10}
                step={0.5}
                value={interpDuration}
                onChange={(e) => setInterpDuration(Number(e.target.value))}
                className="interp-slider"
              />
            </label>
            <button
              className="interpolate-btn"
              onClick={runInterpolation}
              disabled={!canInterpolate || interpLoading}
            >
              {interpLoading ? "Generating…" : "Interpolate"}
            </button>
          </div>
        </div>

        <div className="timeline-scroll" ref={scrollRef}>
          <div className="timeline-ruler" style={{ width: `${timelineWidth}px` }}>
            {Array.from({ length: rulerSeconds }, (_, s) => (
              <div key={s} className="timeline-ruler-mark" style={{ left: `${s * PX_PER_SEC}px` }}>
                <div className="timeline-ruler-tick" />
                <span className="timeline-ruler-label">{s}s</span>
              </div>
            ))}
          </div>

          <div
            className="timeline"
            style={{ width: `${timelineWidth}px` }}
            onDragOver={handleTimelineDragOver}
            onDragLeave={(e) => {
              if (!e.currentTarget.contains(e.relatedTarget as Node)) stopAutoScroll();
            }}
            onDrop={(e) => {
              resetDragState();
              const sound = dragSoundRef.current;
              if (!sound) return;
              const timelineLeft = e.currentTarget.getBoundingClientRect().left;
              const dropX = e.clientX - timelineLeft;
              const newClip: TimelineClip = {
                id: Date.now(),
                name: sound.name,
                filename: sound.filename,
                start: Math.max(0, dropX - CLIP_WIDTH_PX / 2),
                duration: CLIP_WIDTH_PX,
              };
              setTimelineClips((prev) => [...prev, newClip]);
            }}
          >
            {timelineClips.length === 0 && (
              <div className="timeline-empty-hint">Drag sounds here to build the timeline</div>
            )}

            {timelineClips.map((clip) => (
              <div
                key={clip.id}
                className={`timeline-clip${draggingId === clip.id ? " dragging" : ""}${overlaps.includes(clip.id) ? " overlap" : ""}`}
                draggable
                onDragStart={() => { setDraggingId(clip.id); setDragStartWidth(timelineWidth); }}
                onDragEnd={() => { setDraggingId(null); resetDragState(); }}
                onDrag={(e) => {
                  if (e.clientX <= 0) return;
                  const timelineLeft = e.currentTarget.parentElement?.getBoundingClientRect().left ?? 0;
                  moveClip(clip.id, e.clientX - timelineLeft - CLIP_WIDTH_PX / 2);
                }}
                onClick={() => {
                  setSelectedSound(sounds.find((s) => s.filename === clip.filename) ?? null);
                }}
                style={{ left: `${clip.start}px`, width: `${clip.duration}px` }}
              >
                {clip.name}
                <button
                  className="delete-clip-btn"
                  onClick={(e) => { e.stopPropagation(); deleteClip(clip.id); }}
                  title="Remove"
                >✕</button>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
