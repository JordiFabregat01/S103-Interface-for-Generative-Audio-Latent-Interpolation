import { useEffect, useState } from "react";
import "../App.css";

type SoundPoint = {
  id: number;
  name: string;
  filename: string;
  x: number;
  y: number;
};

const API_BASE = "http://localhost:8000";

export default function WorkspaceView() {
  const [sounds, setSounds] = useState<SoundPoint[]>([]);
  const [timelineSounds, setTimelineSounds] = useState<SoundPoint[]>([]);

  useEffect(() => {
    fetch(`${API_BASE}/sounds`)
      .then((res) => res.json())
      .then((data) => setSounds(data))
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

  const handleDragStart = (
    e: React.DragEvent<HTMLDivElement>,
    sound: SoundPoint
  ) => {
    e.dataTransfer.setData("sound", JSON.stringify(sound));
  };

  const handleDropOnTimeline = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();

    const soundData = e.dataTransfer.getData("sound");
    if (!soundData) return;

    const sound: SoundPoint = JSON.parse(soundData);
    setTimelineSounds((prev) => [...prev, sound]);
  };

  return (
    <div className="workspace-page">
      <h1>Workspace</h1>

      <div className="workspace-main">
        <div className="library-panel">
          <h2>Sound Library</h2>

          <div className="sound-grid">
            {sounds.map((sound) => (
              <div
                key={sound.id}
                className="sound-card"
                draggable
                onDragStart={(e) => handleDragStart(e, sound)}
              >
                <div className="sound-image">{getEmoji(sound.name)}</div>
                <p>{sound.name}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="parameters-panel">
          <h2>Parameters</h2>

          <label>
            Interpolation strength
            <input type="range" min="0" max="100" defaultValue="50" />
          </label>

          <label>
            Duration
            <input type="number" defaultValue="5" min="1" />
          </label>

          <label>
            Smoothness
            <input type="range" min="0" max="100" defaultValue="70" />
          </label>

          <button>Generate Audio</button>
        </div>

        <div className="timeline-panel">
          <h2>Timeline</h2>

          <div
            className="timeline"
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDropOnTimeline}
          >
            {timelineSounds.length === 0 ? (
              <span>Drag sounds here to build your interpolation timeline</span>
            ) : (
              timelineSounds.map((sound, index) => (
                <div key={`${sound.id}-${index}`} className="timeline-item">
                  <span>{getEmoji(sound.name)}</span>
                  <p>{sound.name}</p>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
