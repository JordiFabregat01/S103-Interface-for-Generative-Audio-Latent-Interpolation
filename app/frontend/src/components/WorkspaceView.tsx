import { useEffect, useRef, useState } from "react";
import "../App.css";
import { getSounds, getSoundUrl, interpolate, type SoundPoint } from "../api";
import { useAudioPlayer } from "../hooks/useAudioPlayer";

type TimelineClip = {
  id: number;
  name: string;
  filename: string;
  start: number;
  duration: number;
  color: string;
};

const COLORS = ["#3b82f6", "#8b5cf6", "#06b6d4", "#10b981", "#f59e0b", "#ef4444"];

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
  const previewPlayer = useAudioPlayer();
  const interpPlayer = useAudioPlayer();
  const dragSoundRef = useRef<SoundPoint | null>(null);

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
  timelineClips.forEach((clip, index) => {
    const next = timelineClips[index + 1];
    if (next && clip.start < next.start + next.duration && clip.start + clip.duration > next.start) {
      overlaps.push(clip.id, next.id);
    }
  });

  const canInterpolate = timelineClips.length >= 2;

  return (
    <div className="workspace-page">
      <h1>Workspace</h1>

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
                  onDragEnd={() => { dragSoundRef.current = null; }}
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
          <h2>Interpolation Area</h2>
          <div className="drop-zone">Drag sounds here</div>
        </div>
      </div>

      <div className="timeline-panel">
        <h2>Timeline</h2>

        <div className="timeline-controls">
          <label className="duration-label">
            Interpolation duration: <strong>{interpDuration.toFixed(1)}s</strong>
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

        {interpError && <p className="interp-error">{interpError}</p>}

        {interpUrl && !interpLoading && (
          <div className="interp-result">
            <span>Result ready</span>
            <div className="preview-controls">
              <button
                className="preview-play-btn"
                onClick={() => interpPlayer.isPlaying ? interpPlayer.pause() : (previewPlayer.pause(), interpPlayer.play(interpUrl))}
              >
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
          </div>
        )}

        <div
          className="timeline"
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            const sound = dragSoundRef.current;
            if (!sound) return;
            const timelineLeft = e.currentTarget.getBoundingClientRect().left;
            const dropX = e.clientX - timelineLeft;
            const newClip: TimelineClip = {
              id: Date.now(),
              name: sound.name,
              filename: sound.filename,
              start: Math.max(0, dropX - 110),
              duration: 220,
              color: COLORS[timelineClips.length % COLORS.length],
            };
            setTimelineClips((prev) => [...prev, newClip]);
          }}
        >
          {timelineClips.map((clip) => (
            <div
              key={clip.id}
              className={`timeline-clip${draggingId === clip.id ? " dragging" : ""}${overlaps.includes(clip.id) ? " overlap" : ""}`}
              draggable
              onDragStart={() => setDraggingId(clip.id)}
              onDragEnd={() => setDraggingId(null)}
              onDrag={(e) => {
                if (e.clientX <= 0) return;
                const timelineLeft = e.currentTarget.parentElement?.getBoundingClientRect().left || 0;
                moveClip(clip.id, e.clientX - timelineLeft - 100);
              }}
              onClick={() => {
                setSelectedSound(sounds.find((s) => s.filename === clip.filename) ?? null);
                previewPlayer.play(getSoundUrl(clip.filename));
              }}
              style={{ left: `${clip.start}px`, width: `${clip.duration}px`, background: clip.color }}
            >
              <span>{clip.name}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
