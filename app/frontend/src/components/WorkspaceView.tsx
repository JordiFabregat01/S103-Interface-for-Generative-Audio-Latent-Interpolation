import { useEffect, useState } from "react";
import "../App.css";

type SoundPoint = {
  id: number;
  name: string;
  filename: string;
  x: number;
  y: number;
};

type TimelineClip = {
  id: number;
  name: string;
  filename: string;
  start: number;
  duration: number;
  color: string;
};

const API_BASE = "http://localhost:8000";

const audioRef = new Audio();

export default function WorkspaceView() {
  const [sounds, setSounds] = useState<SoundPoint[]>([]);
  const [timelineClips, setTimelineClips] = useState<TimelineClip[]>([]);
  const [draggingId, setDraggingId] = useState<number | null>(null);

  useEffect(() => {
    fetch(`${API_BASE}/sounds`)
      .then((res) => res.json())
      .then((data) => {
        setSounds(data);
      })
      .catch((err) => {
        console.error("Error loading sounds:", err);
      });
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

  const playSound = (filename: string) => {
    audioRef.src = `${API_BASE}/sounds/${encodeURIComponent(filename)}`;
    audioRef.play();
  };

  const playClip = (filename: string) => {
    return new Promise<void>((resolve) => {
      audioRef.src = `${API_BASE}/sounds/${encodeURIComponent(filename)}`;

      audioRef.onended = () => {
        resolve();
      };

      audioRef.play();
    });
  };

  const playTimeline = async () => {
    for (const clip of timelineClips) {
      await playClip(clip.filename);
    }
  };

  const addToTimeline = (sound: SoundPoint) => {
    const lastClip = timelineClips[timelineClips.length - 1];

    const colors = [
      "#3b82f6",
      "#8b5cf6",
      "#06b6d4",
      "#10b981",
      "#f59e0b",
      "#ef4444",
    ];

    const newClip: TimelineClip = {
      id: Date.now(),
      name: sound.name,
      filename: sound.filename,
      start: lastClip
        ? lastClip.start + lastClip.duration - 40
        : 0,
      duration: 220,
      color: colors[timelineClips.length % colors.length],
    };

    setTimelineClips([...timelineClips, newClip]);
  };

  const moveClip = (id: number, newStart: number) => {
    setTimelineClips((prev) =>
      prev.map((clip) =>
        clip.id === id
          ? {
              ...clip,
              start: Math.max(0, newStart),
            }
          : clip
      )
    );
  };

  const hasOverlap = (
    clipA: TimelineClip,
    clipB: TimelineClip
  ) => {
    return (
      clipA.start < clipB.start + clipB.duration &&
      clipA.start + clipA.duration > clipB.start
    );
  };

  const getOverlapAmount = (
    clipA: TimelineClip,
    clipB: TimelineClip
  ) => {
    const overlapStart = Math.max(
      clipA.start,
      clipB.start
    );

    const overlapEnd = Math.min(
      clipA.start + clipA.duration,
      clipB.start + clipB.duration
    );

    const overlap = overlapEnd - overlapStart;

    if (overlap <= 0) return 0;

    return overlap / Math.min(
      clipA.duration,
      clipB.duration
    );
  };

  const overlaps: number[] = [];

  timelineClips.forEach((clip, index) => {
    const nextClip = timelineClips[index + 1];

    if (nextClip && hasOverlap(clip, nextClip)) {
      overlaps.push(clip.id);
      overlaps.push(nextClip.id);

      const amount = getOverlapAmount(
        clip,
        nextClip
      );

      console.log(
        `${clip.name} + ${nextClip.name} = ${amount.toFixed(2)}`
      );
    }
  });

  return (
    <div className="workspace-page">
      <h1>Workspace</h1>

      <div className="workspace-layout">
        {/* LIBRARY */}
        <div className="library-panel">
          <h2>Sound Library</h2>

          <div className="sound-grid">
            {sounds.map((sound) => (
              <div
                key={sound.id}
                className="sound-card"
                onClick={() => addToTimeline(sound)}
              >
                <div className="sound-image">
                  {getEmoji(sound.name)}
                </div>

                <p>{sound.name}</p>
              </div>
            ))}
          </div>
        </div>

        {/* INTERPOLATION AREA */}
        <div className="drop-panel">
          <h2>Interpolation Area</h2>

          <div className="drop-zone">
            Drag sounds here
          </div>
        </div>
      </div>

      {/* TIMELINE */}
      <div className="timeline-panel">
        <h2>Timeline</h2>

        <button
          className="play-timeline-btn"
          onClick={playTimeline}
        >
          ▶ Play Timeline
        </button>

        <div className="timeline">
          {timelineClips.map((clip) => (
            <div
              key={clip.id}
              className={`timeline-clip
                ${draggingId === clip.id ? "dragging" : ""}
                ${overlaps.includes(clip.id) ? "overlap" : ""}
              `}
              draggable
              onDragStart={() => {
                setDraggingId(clip.id);
              }}
              onDragEnd={() => {
                setDraggingId(null);
              }}
              onDrag={(e) => {
                if (e.clientX <= 0) return;

                const timelineLeft =
                  e.currentTarget.parentElement?.getBoundingClientRect().left || 0;

                const newStart = e.clientX - timelineLeft - 100;

                moveClip(clip.id, newStart);
              }}
              onClick={() => playSound(clip.filename)}
              style={{
                left: `${clip.start}px`,
                width: `${clip.duration}px`,
                background: clip.color,
              }}
            >
              <span>{clip.name}</span>

              <button
                className="delete-clip-btn"
                onClick={(e) => {
                  e.stopPropagation();

                  setTimelineClips((prev) =>
                    prev.filter((c) => c.id !== clip.id)
                  );
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