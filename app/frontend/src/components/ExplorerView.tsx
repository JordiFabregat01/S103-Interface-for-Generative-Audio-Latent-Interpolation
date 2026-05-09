import { useEffect, useMemo, useState } from "react";
import "../App.css";
import { getSounds, getSoundUrl, type SoundPoint } from "../api";
import AudioPlayer from "./AudioPlayer";
import { useAudioPlayer } from "../hooks/useAudioPlayer";

const PLOT_WIDTH = 800;
const PLOT_HEIGHT = 500;
const PLOT_PADDING = 40;

export default function ExplorerView() {
  const [points, setPoints] = useState<SoundPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<SoundPoint | null>(null);
  const player = useAudioPlayer();

  useEffect(() => {
    let cancelled = false;

    getSounds()
      .then((data) => {
        if (cancelled) return;
        setPoints(data);
        setLoading(false);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err.message ?? "Failed to load sounds");
        setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const placedPoints = useMemo(() => {
    if (points.length === 0) return [];

    const innerW = PLOT_WIDTH - 2 * PLOT_PADDING;
    const innerH = PLOT_HEIGHT - 2 * PLOT_PADDING;

    return points.map((p) => ({
      ...p,
      px: PLOT_PADDING + p.x * innerW,
      py: PLOT_PADDING + p.y * innerH,
    }));
  }, [points]);

  return (
    <div className="app">
      <header className="header">
        <h1>Generative Audio Interface</h1>
        <p>Explore latent sound space (CLAP + t-SNE)</p>
      </header>

      <main className="main-layout">
        <div
          className="plot-area"
          style={{ width: PLOT_WIDTH, height: PLOT_HEIGHT }}
        >
          {loading && <div className="plot-message">Loading sounds…</div>}

          {error && (
            <div className="plot-message error">
              Failed to load sounds: {error}
            </div>
          )}

          {!loading &&
            !error &&
            placedPoints.map((point) => {
              const isSelected = selected?.id === point.id;

              return (
                <div
                  key={point.id}
                  className={`dot${isSelected ? " selected" : ""}`}
                  style={{ left: point.px, top: point.py }}
                  onClick={() => {
                    if (selected?.id === point.id) {
                      player.isPlaying ? player.pause() : player.play(getSoundUrl(point.filename));
                    } else {
                      setSelected(point);
                      player.play(getSoundUrl(point.filename));
                    }
                  }}
                  title={point.name}
                >
                  <span className="dot-marker">●</span>
                  <span className="dot-label">{point.name}</span>
                </div>
              );
            })}
        </div>

        <div className="side-panel">
          <h2>Sound Info</h2>

          {selected ? (
            <>
              <p>
                <strong>Name:</strong> {selected.name}
              </p>

              <p>
                <strong>File:</strong> {selected.filename}
              </p>

              <p>
                t-SNE: ({selected.x.toFixed(3)},{" "}
                {selected.y.toFixed(3)})
              </p>

              <AudioPlayer player={player} url={getSoundUrl(selected.filename)} />
              <button>Use for Interpolation</button>
            </>
          ) : (
            <p>Select a point to inspect a sound.</p>
          )}
        </div>
      </main>
    </div>
  );
}