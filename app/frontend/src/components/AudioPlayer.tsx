import { useEffect, useRef } from "react";
import type { AudioPlayerControls } from "../hooks/useAudioPlayer";

type Props = {
  player: AudioPlayerControls;
  url: string;
};

function fmt(sec: number) {
  if (!isFinite(sec)) return "0:00";
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export default function AudioPlayer({ player, url }: Props) {
  const { isPlaying, currentTime, duration, play, pause, seek, load } = player;
  const prevUrl = useRef(url);

  useEffect(() => {
    if (url === prevUrl.current) return;
    prevUrl.current = url;
    if (isPlaying) {
      play(url);
    } else {
      load(url);
    }
  }, [url]);

  const toggle = () => (isPlaying ? pause() : play(url));
  const restart = () => { seek(0); play(url); };

  return (
    <div className="audio-player">
      <button className="audio-play-btn" onClick={restart} title="Restart">↺</button>
      <button className="audio-play-btn" onClick={toggle}>
        {isPlaying ? "⏸" : "▶"}
      </button>

      <input
        className="audio-scrubber"
        type="range"
        min={0}
        max={duration || 1}
        step={0.01}
        value={currentTime}
        onChange={(e) => seek(Number(e.target.value))}
      />

      <span className="audio-time">
        {fmt(currentTime)} / {fmt(duration)}
      </span>
    </div>
  );
}
