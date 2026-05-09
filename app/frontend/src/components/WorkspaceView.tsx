import { useEffect, useRef, useState } from "react";
import "../App.css";
import { getSounds, getSoundUrl, type SoundPoint } from "../api";
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

function fmt(sec: number) {
  if (!isFinite(sec)) return "0:00";
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export default function WorkspaceView() {
  const [sounds, setSounds] = useState<SoundPoint[]>([]);
  const [timelineClips, setTimelineClips] = useState<TimelineClip[]>([]);
  const [draggingId, setDraggingId] = useState<number | null>(null);
  const [selectedSound, setSelectedSound] = useState<SoundPoint | null>(null);
  const player = useAudioPlayer();
  const dragSoundRef = useRef<SoundPoint | null>(null);

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

  const addToTimeline = (sound: SoundPoint) => {
    const lastClip = timelineClips[timelineClips.length - 1];
    const newClip: TimelineClip = {
      id: Date.now(),
      name: sound.name,
      filename: sound.filename,
      start: lastClip ? lastClip.start + lastClip.duration - 40 : 0,
      duration: 220,
      color: COLORS[timelineClips.length % COLORS.length],
    };
    setTimelineClips((prev) => [...prev, newClip]);
  };

  const moveClip = (id: number, newStart: number) => {
    setTimelineClips((prev) =>
      prev.map((clip) =>
        clip.id === id ? { ...clip, start: Math.max(0, newStart) } : clip
      )
    );
  };

  const playTimeline = async () => {
    for (const clip of timelineClips) {
      await new Promise<void>((resolve) => {
        const audio = new Audio(getSoundUrl(clip.filename));
        audio.onended = () => resolve();
        audio.play();
      });
    }
  };

  const overlaps: number[] = [];
  timelineClips.forEach((clip, index) => {
    const next = timelineClips[index + 1];
    if (next && clip.start < next.start + next.duration && clip.start + clip.duration > next.start) {
      overlaps.push(clip.id, next.id);
    }
  });

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
                      player.pause();
                      setSelectedSound(null);
                    } else {
                      setSelectedSound(sound);
                      player.play(getSoundUrl(sound.filename));
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
                <button className="close-preview-btn" onClick={() => { player.pause(); setSelectedSound(null); }}>✕</button>
              </div>
              <div className="preview-controls">
                <button
                  className="preview-play-btn"
                  onClick={() => { player.seek(0); player.play(getSoundUrl(selectedSound.filename)); }}
                  title="Restart"
                >↺</button>
                <button
                  className="preview-play-btn"
                  onClick={() => player.isPlaying ? player.pause() : player.play(getSoundUrl(selectedSound.filename))}
                >
                  {player.isPlaying ? "⏸" : "▶"}
                </button>
                <input
                  className="audio-scrubber"
                  type="range"
                  min={0}
                  max={player.duration || 1}
                  step={0.01}
                  value={player.currentTime}
                  onChange={(e) => player.seek(Number(e.target.value))}
                />
                <span className="audio-time">{fmt(player.currentTime)} / {fmt(player.duration)}</span>
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

        <button className="play-timeline-btn" onClick={playTimeline}>
          ▶ Play Timeline
        </button>

        <div
          className="timeline"
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            const sound = dragSoundRef.current;
            if (!sound) return;
            const timelineLeft = e.currentTarget.getBoundingClientRect().left;
            const dropX = e.clientX - timelineLeft;
            const lastClip = timelineClips[timelineClips.length - 1];
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
                player.play(getSoundUrl(clip.filename));
              }}
              style={{ left: `${clip.start}px`, width: `${clip.duration}px`, background: clip.color }}
            >
              <span>{clip.name}</span>
              <button
                className="delete-clip-btn"
                onClick={(e) => {
                  e.stopPropagation();
                  setTimelineClips((prev) => prev.filter((c) => c.id !== clip.id));
                }}
              >
                ✕
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
