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

  return (
    <div className="workspace-page">
      <h1>Workspace</h1>

      <div className="workspace-layout">

        {/* LIBRARY */}
        <div className="library-panel">
          <h2>Sound Library</h2>

          <div className="sound-grid">
            {sounds.map((sound) => (
              <div key={sound.id} className="sound-card">
                <div className="sound-image">
                  {getEmoji(sound.name)}
                </div>

                <p>{sound.name}</p>
              </div>
            ))}
          </div>
        </div>

        {/* DROP ZONE */}
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

        <div className="timeline">
          Timeline interpolation system coming soon...
        </div>
      </div>
    </div>
  );
}