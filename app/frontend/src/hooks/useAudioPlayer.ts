import { useEffect, useRef, useState } from "react";

export type AudioPlayerControls = {
  isPlaying: boolean;
  currentTime: number;
  duration: number;
  play: (url: string) => void;
  pause: () => void;
  seek: (time: number) => void;
  load: (url: string) => void;
};

export function useAudioPlayer(): AudioPlayerControls {
  const audioRef = useRef<HTMLAudioElement>(new Audio());
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  useEffect(() => {
    const audio = audioRef.current;

    const onPlay = () => setIsPlaying(true);
    const onPause = () => setIsPlaying(false);
    const onEnded = () => { setIsPlaying(false); setCurrentTime(0); };
    const onTimeUpdate = () => setCurrentTime(audio.currentTime);
    const onDurationChange = () => setDuration(audio.duration || 0);

    audio.addEventListener("play", onPlay);
    audio.addEventListener("pause", onPause);
    audio.addEventListener("ended", onEnded);
    audio.addEventListener("timeupdate", onTimeUpdate);
    audio.addEventListener("durationchange", onDurationChange);

    return () => {
      audio.removeEventListener("play", onPlay);
      audio.removeEventListener("pause", onPause);
      audio.removeEventListener("ended", onEnded);
      audio.removeEventListener("timeupdate", onTimeUpdate);
      audio.removeEventListener("durationchange", onDurationChange);
      audio.pause();
    };
  }, []);

  const play = (url: string) => {
    const audio = audioRef.current;
    if (audio.src !== url) {
      audio.src = url;
      setCurrentTime(0);
      setDuration(0);
    }
    audio.play().catch(console.error);
  };

  const pause = () => audioRef.current.pause();

  const seek = (time: number) => {
    audioRef.current.currentTime = time;
    setCurrentTime(time);
  };

  const load = (url: string) => {
    const audio = audioRef.current;
    audio.pause();
    audio.src = url;
    setCurrentTime(0);
    setDuration(0);
  };

  return { isPlaying, currentTime, duration, play, pause, seek, load };
}
