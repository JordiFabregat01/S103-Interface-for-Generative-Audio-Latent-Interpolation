import { useState } from "react";
import "./App.css";

type SoundPoint = {
  id: number;
  name: string;
  x: number;
  y: number;
};

const points: SoundPoint[] = [
  { id: 1, name: "Rain", x: 120, y: 100 },
  { id: 2, name: "Fire", x: 300, y: 180 },
  { id: 3, name: "Birds", x: 500, y: 90 },
  { id: 4, name: "Keyboard", x: 250, y: 320 },
  { id: 5, name: "River", x: 600, y: 250 },
];

function App() {
  const [selected, setSelected] = useState<SoundPoint | null>(null);

  return (
    <div className="app">
      <header className="header">
        <h1>Generative Audio Interface</h1>
        <p>Explore latent sound space</p>
      </header>

      <main className="main-layout">
        <div className="plot-area">
          {points.map((point) => (
            <div
              key={point.id}
              className="dot"
              style={{
                left: point.x,
                top: point.y,
              }}
              onClick={() => setSelected(point)}
            >
              ●
            </div>
          ))}
        </div>

        <div className="side-panel">
          <h2>Sound Info</h2>

          {selected ? (
            <>
              <p><strong>Name:</strong> {selected.name}</p>
              <p>X: {selected.x}</p>
              <p>Y: {selected.y}</p>

              <button>Play Sound</button>
              <button>Use for Interpolation</button>
            </>
          ) : (
            <p>Select a point</p>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;