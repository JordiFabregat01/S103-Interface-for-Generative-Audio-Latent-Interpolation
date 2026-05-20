import { useEffect, useMemo, useRef, useState } from "react";
import "../App.css";
import { getSounds, getSoundUrl, searchSounds, findSimilarSounds, render, type Segment, type SoundPoint, type SoundHit } from "../api";
import { useAudioPlayer } from "../hooks/useAudioPlayer";

// start and duration are stored in seconds; pixels = value * pxPerSec
type TimelineClip = {
  id: number;
  name: string;
  filename: string;
  start: number;
  duration: number;
};

type ResizeState = {
  id: number;
  edge: "left" | "right";
  startMouseX: number;
  startSec: number;
  startDur: number;
};

const DEFAULT_PX_PER_SEC = 80;
const MIN_PX_PER_SEC = 20;
const MAX_PX_PER_SEC = 80;
const DEFAULT_CLIP_DURATION_SEC = 3;
const TIMELINE_BUFFER_PX = 400;
const MIN_CLIP_DURATION_SEC = 0.25;
const SNAP_THRESHOLD_PX = 12;
const LOADING_VERBS = ["working", "cooking", "interpolating", "generating"] as const;

function snapEdge(value: number, targets: number[], threshold: number): { snapped: number; dist: number } {
  let best = value;
  let bestDist = threshold;
  for (const t of targets) {
    const d = Math.abs(value - t);
    if (d < bestDist) { bestDist = d; best = t; }
  }
  return { snapped: best, dist: bestDist };
}

function clipEdgeTargets(clips: TimelineClip[], excludeId: number): number[] {
  const targets: number[] = [0];
  for (const c of clips) {
    if (c.id === excludeId) continue;
    targets.push(c.start, c.start + c.duration);
  }
  return targets;
}


function fmt(sec: number) {
  if (!isFinite(sec)) return "0:00";
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function filenameToAudioElement(filename: string) {
  return filename.replace(/\.[^.]+$/, "");
}


export default function WorkspaceView() {
  const [sounds, setSounds] = useState<SoundPoint[]>([]);
  const [timelineClips, setTimelineClips] = useState<TimelineClip[]>([]);
  const [draggingId, setDraggingId] = useState<number | null>(null);
  const [selectedClipId, setSelectedClipId] = useState<number | null>(null);
  const [resizing, setResizing] = useState<ResizeState | null>(null);
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
  const clipsRef = useRef<TimelineClip[]>([]);
  const [quality, setQuality] = useState(8);
  const [showSettings, setShowSettings] = useState(false);
  const [showHowToUse, setShowHowToUse] = useState(false);
  const [showAbout, setShowAbout] = useState(false);
  const [pxPerSec, setPxPerSec] = useState(DEFAULT_PX_PER_SEC);
  const pxPerSecRef = useRef(DEFAULT_PX_PER_SEC);
  const zoomCenterSecRef = useRef<number | null>(null);
  useEffect(() => { pxPerSecRef.current = pxPerSec; }, [pxPerSec]);
  useEffect(() => {
    if (zoomCenterSecRef.current === null) return;
    const scroll = scrollRef.current;
    if (!scroll) return;
    scroll.scrollLeft = zoomCenterSecRef.current * pxPerSec - scroll.clientWidth / 2;
    zoomCenterSecRef.current = null;
  }, [pxPerSec]);
  useEffect(() => { clipsRef.current = timelineClips; }, [timelineClips]);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      if (!e.ctrlKey) return;
      e.preventDefault();
      const centerSec = (el.scrollLeft + el.clientWidth / 2) / pxPerSecRef.current;
      zoomCenterSecRef.current = centerSec;
      setPxPerSec((p) =>
        e.deltaY < 0
          ? Math.min(MAX_PX_PER_SEC, p * 2)
          : Math.max(MIN_PX_PER_SEC, p / 2)
      );
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  useEffect(() => {
    if (interpolatedGaps.size === 0) return;
    const sorted = [...timelineClips].sort((a, b) => a.start - b.start);
    const validKeys = new Set(sorted.flatMap((clip, i) => {
      const next = sorted[i + 1];
      if (!next) return [];
      const gap = next.start - (clip.start + clip.duration);
      return gap > 0.01 ? [`${clip.id}-${next.id}`] : [];
    }));
    setInterpolatedGaps((prev) => {
      const cleaned = new Set([...prev].filter((k) => validKeys.has(k)));
      return cleaned.size === prev.size ? prev : cleaned;
    });
  }, [timelineClips]);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => setContainerWidth(entry.contentRect.width));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Resize handle drag via mouse events (more reliable than HTML drag for edge resizing)
  useEffect(() => {
    if (!resizing) return;
    const onMove = (e: MouseEvent) => {
      const dx = (e.clientX - resizing.startMouseX) / pxPerSecRef.current;
      setTimelineClips((prev) => {
        const targets = clipEdgeTargets(prev, resizing.id);
        return prev.map((c) => {
          if (c.id !== resizing.id) return c;
          const snapThreshold = SNAP_THRESHOLD_PX / pxPerSecRef.current;
          if (resizing.edge === "right") {
            const rawRight = resizing.startSec + resizing.startDur + dx;
            const { snapped } = snapEdge(rawRight, targets, snapThreshold);
            return { ...c, duration: Math.max(MIN_CLIP_DURATION_SEC, snapped - resizing.startSec) };
          } else {
            const rawLeft = resizing.startSec + dx;
            const { snapped } = snapEdge(rawLeft, targets, snapThreshold);
            const newStart = Math.max(0, snapped);
            const moved = newStart - resizing.startSec;
            return {
              ...c,
              start: newStart,
              duration: Math.max(MIN_CLIP_DURATION_SEC, resizing.startDur - moved),
            };
          }
        });
      });
      setInterpUrl("");
    };
    const onUp = () => setResizing(null);
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
    return () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
  }, [resizing]);

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

  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<SoundHit[] | null>(null);
  const [searchLoading, setSearchLoading] = useState(false);
  const [similarTo, setSimilarTo] = useState<SoundPoint | null>(null);
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; sound: SoundPoint } | null>(null);

  useEffect(() => {
    if (!contextMenu) return;
    const dismiss = () => setContextMenu(null);
    document.addEventListener("mousedown", dismiss);
    return () => document.removeEventListener("mousedown", dismiss);
  }, [contextMenu]);

  useEffect(() => {
    const q = searchQuery.trim();
    if (!q) { setSearchResults(null); setSimilarTo(null); return; }
    const timer = setTimeout(() => {
      setSearchLoading(true);
      searchSounds(q)
        .then(setSearchResults)
        .catch(() => setSearchResults([]))
        .finally(() => setSearchLoading(false));
    }, 300);
    return () => clearTimeout(timer);
  }, [searchQuery]);

  const [interpLoading, setInterpLoading] = useState(false);
  const [interpError, setInterpError] = useState<string | null>(null);
  const [interpUrl, setInterpUrl] = useState<string | null>(null);
  const interpUrlRef = useRef<string | null>(null);
  const [interpolatedGaps, setInterpolatedGaps] = useState<Set<string>>(new Set());
  const [loadingVerbIdx, setLoadingVerbIdx] = useState(0);

  useEffect(() => {
    if (!interpLoading) {
      setLoadingVerbIdx(0);
      return;
    }
    const id = setInterval(() => {
      setLoadingVerbIdx((i) => (i + 1) % LOADING_VERBS.length);
    }, 1400 * 3);
    return () => clearInterval(id);
  }, [interpLoading]);

  useEffect(() => {
    getSounds()
      .then(setSounds)
      .catch((err) => console.error("Error loading sounds:", err));
  }, []);
const getSoundColor = (name: string, filename = ""): { color: string; glow: string } => {
  const lower = `${name} ${filename}`.toLowerCase();
  if (lower.includes("wind") || lower.includes("breeze"))
    return { color: "#6ee7d4", glow: "rgba(110, 231, 212, 0.45)" };
  if (lower.includes("thunder") || lower.includes("storm"))
    return { color: "#a78bfa", glow: "rgba(167, 139, 250, 0.45)" };
  if (lower.includes("rain"))
    return { color: "#7dd3fc", glow: "rgba(125, 211, 252, 0.45)" };
  if (lower.includes("waterfall") || lower.includes("waterrocks") || lower.includes("underwater") || lower.includes("river") || lower.includes("sea") || lower.includes("waves") || lower.includes("water"))
    return { color: "#38bdf8", glow: "rgba(56, 189, 248, 0.45)" };
  if (lower.includes("fire"))
    return { color: "#fb923c", glow: "rgba(251, 146, 60, 0.45)" };
  if (lower.includes("bird") || lower.includes("seagull") || lower.includes("loon"))
    return { color: "#86efac", glow: "rgba(134, 239, 172, 0.45)" };
  if (lower.includes("bee") || lower.includes("cicada") || lower.includes("cricket"))
    return { color: "#fde047", glow: "rgba(253, 224, 71, 0.45)" };
  if (lower.includes("footstep") || lower.includes("keyboard") || lower.includes("step"))
    return { color: "#c084fc", glow: "rgba(192, 132, 252, 0.45)" };
  return { color: "#58a6ff", glow: "rgba(88, 166, 255, 0.45)" };
};

const getEmoji = (name: string, filename = "") => {
  const lower = `${name} ${filename}`.toLowerCase();

  // WIND
  if (lower.includes("articwind")) return "🌬️";
  if (lower.includes("cornfieldwind")) return "🌾💨";
  if (lower.includes("Intense")) return "🍃💨";
  if (lower.includes("breeze")) return "🍃";
  if (lower.includes("wind")) return "💨";

  // RAIN / STORM
  if (lower.includes("icestorm")) return "🧊⛈️";
  if (lower.includes("thunder")) return "⚡";
  if (lower.includes("storm")) return "⛈️";
  if (lower.includes("heavyrain")) return "🌧️💧";
  if (lower.includes("rain")) return "☔";

  // WATER
  if (lower.includes("waterfall")) return "💦⬇️";
  if (lower.includes("waterrocks")) return "🫧🪨";
  if (lower.includes("underwater")) return "🐠";
  if (lower.includes("slowriver")) return "🏞️";
  if (lower.includes("river")) return "🫧";
  if (lower.includes("waves")) return "🌊〰️";
  if (lower.includes("sea")) return "🌊";
  if (lower.includes("water")) return "💧";

  // FIRE
  if (lower.includes("camp_fire")) return "🔥🌲";
  if (lower.includes("fire")) return "🔥";

  // BIRDS / ANIMALS
  if (lower.includes("seagull")) return "🕊️";
  if (lower.includes("loon")) return "🌙";
  if (lower.includes("bird")) return "🐦";

  // INSECTS
  if (lower.includes("bees")) return "🐝";
  if (lower.includes("cicada")) return "🪲";
  if (lower.includes("cricket")) return "🦗";

  // HUMAN
  if (lower.includes("snowsteps")) return "🥾❄️";
  if (lower.includes("footsteps")) return "👣";
  if (lower.includes("keyboard")) return "⌨️";

  return "🎵";
};

const moveClip = (id: number, newStartSec: number) => {
  setTimelineClips((prev) =>
    prev.map((clip) =>
      clip.id === id ? { ...clip, start: Math.max(0, newStartSec) } : clip
    )
  );

  setInterpUrl("");
};

  // Turn the placed clips + gaps into a /render timeline: a leading silence for
  // the start offset, each clip, and between consecutive clips either an
  // interpolation (touching/overlapping, or a gap the user toggled on) or
  // silence (an untouched gap).
  const buildTimelineSegments = (sorted: TimelineClip[]): Segment[] => {
    const segments: Segment[] = [];
    const first = sorted[0];
    if (first && first.start > 0.01) {
      segments.push({ type: "silence", duration: first.start });
    }
    sorted.forEach((clip, index) => {
      segments.push({ type: "clip", filename: clip.filename, duration: clip.duration });
      const next = sorted[index + 1];
      if (!next) return;
      const distanceSec = next.start - (clip.start + clip.duration);
      if (distanceSec > 0.01) {
        const key = `${clip.id}-${next.id}`;
        if (interpolatedGaps.has(key)) {
          segments.push({
            type: "interpolation",
            audio1: filenameToAudioElement(clip.filename),
            audio2: filenameToAudioElement(next.filename),
            distance_sec: distanceSec,
            nfe: quality,
          });
        } else {
          segments.push({ type: "silence", duration: distanceSec });
        }
      } else {
        // Touching (distance ~ 0) or overlapping (distance < 0): crossfade.
        const adjacent = Math.abs(distanceSec) <= 0.01;
        segments.push({
          type: "interpolation",
          audio1: filenameToAudioElement(clip.filename),
          audio2: filenameToAudioElement(next.filename),
          distance_sec: adjacent ? 0 : distanceSec,
          nfe: quality,
          ...(adjacent ? { duration_sec: Math.min(clip.duration, next.duration) } : {}),
        });
      }
    });
    return segments;
  };

  const runInterpolation = async () => {
    const sorted = [...timelineClips].sort((a, b) => a.start - b.start);
    if (sorted.length < 2) return;

    setInterpLoading(true);
    setInterpError(null);
    if (interpUrlRef.current) {
      URL.revokeObjectURL(interpUrlRef.current);
      interpUrlRef.current = null;
    }
    setInterpUrl(null);
    try {
      const url = await render(buildTimelineSegments(sorted));
      interpUrlRef.current = url;
      setInterpUrl(url);
    } catch (err) {
      setInterpError(err instanceof Error ? err.message : "Render failed");
    } finally {
      setInterpLoading(false);
    }
  };

  const sortedClips = [...timelineClips].sort((a, b) => a.start - b.start);

  const overlapRegions: { start: number; end: number }[] = [];
  const gapRegions: { start: number; end: number; clipA: TimelineClip; clipB: TimelineClip; key: string }[] = [];
  sortedClips.forEach((clip, index, arr) => {
    const next = arr[index + 1];
    if (!next) return;
    const clipEnd = clip.start + clip.duration;
    if (clipEnd > next.start) {
      overlapRegions.push({
        start: next.start,
        end: Math.min(clipEnd, next.start + next.duration),
      });
    } else if (next.start - clipEnd > 0.01) {
      gapRegions.push({ start: clipEnd, end: next.start, clipA: clip, clipB: next, key: `${clip.id}-${next.id}` });
    }
  });

  // Junction positions (px) where two clips are exactly touching (snapped)
  const snapJoints: number[] = [];
  sortedClips.forEach((clip, index, arr) => {
    const next = arr[index + 1];
    if (next && Math.abs(next.start - (clip.start + clip.duration)) < 0.01) {
      snapJoints.push((clip.start + clip.duration) * pxPerSec);
    }
  });

  const rightmostSec = timelineClips
    .filter((c) => c.id !== draggingId)
    .reduce((max, c) => Math.max(max, c.start + c.duration), 0);
  const rightmostPx = rightmostSec * pxPerSec;
  const snapUp = (px: number) => Math.ceil(px / pxPerSec) * pxPerSec;
  const clipWidth = rightmostPx > 0 ? snapUp(rightmostPx + TIMELINE_BUFFER_PX) : 0;
  const dragWidth = dragExtentPx > containerWidth ? snapUp(dragExtentPx + TIMELINE_BUFFER_PX) : 0;
  const timelineWidth = Math.max(containerWidth, clipWidth, dragWidth, draggingId ? dragStartWidth : 0);
  const rulerInterval = Math.max(1, DEFAULT_PX_PER_SEC / pxPerSec);
  const rulerMarkCount = Math.floor(timelineWidth / (rulerInterval * pxPerSec));

  const canInterpolate = timelineClips.length >= 2;

  const placedPoints = useMemo(() => sounds.map((p) => ({ ...p, px: p.x * 100, py: p.y * 100 })), [sounds]);

const selectedPathPoints = sortedClips
  .map((clip) =>
    placedPoints.find((point) => point.filename === clip.filename)
  )
  .filter(Boolean);

const selectedPath = selectedPathPoints
  .map((point) => `${point!.px},${point!.py}`)
  .join(" ");

  const deleteClip = (id: number) => {
    setTimelineClips((prev) => prev.filter((c) => c.id !== id));
    setInterpUrl("");
    if (selectedClipId === id) setSelectedClipId(null);
  };

  return (
    <div className="workspace-page">
      <div className="app-header">
        <h1><span style={{ color: "#1E90FF" }}>GALI</span> Generative Audio Latent Interpolation</h1>

        <div className="app-header-actions">
          <button
            className="settings-btn help-btn"
            onClick={() => { setShowAbout(false); setShowHowToUse(!showHowToUse); }}
            title="How to use"
          >
            ?
          </button>

          <button
            className="settings-btn"
            onClick={() => { setShowHowToUse(false); setShowAbout(!showAbout); }}
            title="About"
          >
            About
          </button>

          <button
            className="settings-btn"
            onClick={() => setShowSettings(!showSettings)}
          >
            ⚙️
          </button>
        </div>

        <img
          className="app-logo"
          src="/GALI.jpeg"
          alt="GALI"
          title="Meow!"
          onClick={() => {
            const meow = new Audio("/meow.wav");
            meow.play();
          }}
        />
      </div>

      {showSettings && (
      <>
        <div className="settings-backdrop" onClick={() => setShowSettings(false)} />
        <div className="settings-panel">
        <h3>Settings</h3>

        <div className="quality-selector">
        <label>Quality</label>

        <select
          value={quality}
          onChange={(e) => { setQuality(Number(e.target.value)); setShowSettings(false); }}
        >
          <option value={4}>Fast</option>
          <option value={8}>Balanced</option>
          <option value={16}>High</option>
        </select>
      </div>
    </div>
      </>
)}

      {showHowToUse && (
        <>
          <div className="info-modal-backdrop" onClick={() => setShowHowToUse(false)} />
          <div className="info-modal" role="dialog" aria-modal="true" aria-label="How to use">
            <div className="info-modal-header">
              <h3>How to Use GALI</h3>
              <button className="close-preview-btn" onClick={() => setShowHowToUse(false)}>✕</button>
            </div>
            <div className="info-modal-body">
              <div className="how-to-step">
                <span className="how-to-num">1</span>
                <div>
                  <strong>Browse the Sound Library</strong>
                  <p>The left panel lists all available ambient sounds. Type in the search bar to filter by keyword, or use <em>Find Similar</em> on any sound to surface related ones via AI embedding search.</p>
                </div>
              </div>
              <div className="how-to-step">
                <span className="how-to-num">2</span>
                <div>
                  <strong>Preview a Sound</strong>
                  <p>Click any sound card to play it. A mini player appears at the bottom of the library with scrubbing and restart controls. Click again to pause.</p>
                </div>
              </div>
              <div className="how-to-step">
                <span className="how-to-num">3</span>
                <div>
                  <strong>Build Your Timeline</strong>
                  <p>Drag sound cards from the library — or dots from the Latent Space Explorer — onto the timeline. Clips snap to each other's edges. Drag a clip to reposition it; click a clip to select it, then drag the blue handles on its edges to resize.</p>
                </div>
              </div>
              <div className="how-to-step">
                <span className="how-to-num">4</span>
                <div>
                  <strong>Add Interpolations</strong>
                  <p>When two clips <em>touch or overlap</em>, GALI automatically crossfades between them. When there's a <em>gap</em>, click the dashed region between clips to mark it as an interpolation (amber striped) — otherwise it stays silent. Click the ✕ on a gap region to revert it to silence.</p>
                </div>
              </div>
              <div className="how-to-step">
                <span className="how-to-num">5</span>
                <div>
                  <strong>Explore Latent Space</strong>
                  <p>The right panel visualises all sounds as dots in a 2-D latent space. Sounds that are acoustically similar sit closer together. You can preview any dot by clicking it, or drag it directly to the timeline.</p>
                </div>
              </div>
              <div className="how-to-step">
                <span className="how-to-num">6</span>
                <div>
                  <strong>Render &amp; Download</strong>
                  <p>Hit <strong>Interpolate</strong> (bottom-right of the timeline) to render the full sequence. When it's ready, use the player in the timeline header to preview, and click <strong>Download WAV</strong> to save the file.</p>
                </div>
              </div>
              <div className="how-to-tip">
                <strong>Tip:</strong> Use <kbd>Ctrl</kbd> + scroll on the timeline to zoom in or out. The zoom level shows as a percentage next to the −/+ buttons.
              </div>
            </div>
          </div>
        </>
      )}

      {showAbout && (
        <>
          <div className="info-modal-backdrop" onClick={() => setShowAbout(false)} />
          <div className="info-modal" role="dialog" aria-modal="true" aria-label="About">
            <div className="info-modal-header">
              <h3>About GALI</h3>
              <button className="close-preview-btn" onClick={() => setShowAbout(false)}>✕</button>
            </div>
            <div className="info-modal-body about-body">
              <p className="about-tagline"><span style={{ color: "#1E90FF" }}>GALI</span> — Generative Audio Latent Interpolation</p>
              <p>GALI is a tool for exploring and blending ambient soundscapes using generative AI. Sounds are encoded into a shared latent space using the <strong>CLAP</strong> audio-language model, letting you search by text, find acoustically similar sounds, and interpolate between them to create seamless audio transitions.</p>
              <p>Place sounds on the timeline, define where crossfades and interpolations happen, and render a fully blended audio composition all in the browser.</p>
              <div className="about-divider" />
              <p className="about-credits">Built as part of the Music Technology Group at <strong>Universitat Pompeu Fabra</strong></p>
            </div>
          </div>
        </>
      )}

      <div className="workspace-layout">
        <div className="library-panel">
          <h2>Sound Library</h2>
          <p className="library-hint">Click to preview · Drag to timeline</p>

          <div className="library-search">
            <input
              className="library-search-input"
              type="text"
              placeholder="Search sounds…"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
            {searchQuery && (
              <button className="library-search-clear" onClick={() => setSearchQuery("")}>✕</button>
            )}
          </div>

          {similarTo && (
            <div className="similar-banner">
              <span>Similar to: <strong>{similarTo.name}</strong></span>
              <button className="library-search-clear" onClick={() => { setSearchResults(null); setSimilarTo(null); }}>✕</button>
            </div>
          )}

          <div className="sound-grid-scroll">
            <div className="sound-grid">
              {searchLoading && <p className="library-hint">Searching…</p>}
              {!searchLoading && searchResults !== null && !similarTo && searchResults.length === 0 && (
                <p className="library-hint">No results for "{searchQuery}"</p>
              )}
              {(() => {
                const renderCard = (sound: SoundPoint, extra = "") => (
                  <div
                    key={sound.id}
                    className={`sound-card${selectedSound?.id === sound.id ? " selected" : ""}${extra ? ` ${extra}` : ""}`}
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
                    onContextMenu={(e) => { e.preventDefault(); setContextMenu({ x: e.clientX, y: e.clientY, sound }); }}
                    onDragStart={() => { dragSoundRef.current = sound; }}
                    onDragEnd={() => { dragSoundRef.current = null; resetDragState(); }}
                  >
                    <div className="sound-image">{getEmoji(sound.name)}</div>
                    <p>{sound.name}</p>
                  </div>
                );

                if (similarTo && searchResults !== null) {
                  const similarIds = new Set(searchResults.map((s) => s.id));
                  const others = sounds.filter((s) => !similarIds.has(s.id) && s.id !== similarTo.id);
                  return (
                    <>
                      {searchResults.map((s) => renderCard(s, "similar"))}
                      {renderCard(similarTo, "similar-origin")}
                      {others.map((s) => renderCard(s))}
                    </>
                  );
                }

                return (searchResults ?? sounds).map((s) => renderCard(s));
              })()}
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
              <button
                className="find-similar-btn"
                onClick={() => {
                  if (similarTo?.id === selectedSound.id) {
                    setSearchResults(null);
                    setSimilarTo(null);
                    return;
                  }
                  setSearchQuery("");
                  setSimilarTo(selectedSound);
                  setSearchLoading(true);
                  findSimilarSounds(selectedSound.filename)
                    .then(setSearchResults)
                    .catch(() => setSearchResults([]))
                    .finally(() => setSearchLoading(false));
                }}
              >
                {similarTo?.id === selectedSound.id ? "Clear similar" : "Find similar"}
              </button>
            </div>
          )}
        </div>

        {contextMenu && (
          <div
            className="sound-context-menu"
            style={{ top: contextMenu.y, left: contextMenu.x }}
            onMouseDown={(e) => e.stopPropagation()}
          >
            {similarTo?.id === contextMenu.sound.id ? (
              <button onClick={() => { setSearchResults(null); setSimilarTo(null); setContextMenu(null); }}>
                Clear similar
              </button>
            ) : (
              <button onClick={() => {
                setSearchQuery("");
                setSimilarTo(contextMenu.sound);
                setSearchLoading(true);
                setContextMenu(null);
                findSimilarSounds(contextMenu.sound.filename)
                  .then(setSearchResults)
                  .catch(() => setSearchResults([]))
                  .finally(() => setSearchLoading(false));
              }}>
                Find similar
              </button>
            )}
          </div>
        )}

        <div className="drop-panel">
          <h2>Latent Space Exploration</h2>
          <p className="library-hint">Click to preview · Drag to timeline</p>
          <div className="explorer-plot-wrap">
            <div className="explorer-plot">
              {/* Territory blobs */}
              <svg className="explorer-territory-svg" viewBox="0 0 100 100" preserveAspectRatio="none">
                <defs>
                  <filter id="territory-blur" x="-80%" y="-80%" width="260%" height="260%">
                    <feGaussianBlur stdDeviation="5" />
                  </filter>
                </defs>
                {placedPoints.map((point) => {
                  const { color } = getSoundColor(point.name, point.filename);
                  return (
                    <circle
                      key={point.id}
                      cx={point.px}
                      cy={point.py}
                      r="7"
                      fill={color}
                      opacity="0.06"
                      filter="url(#territory-blur)"
                    />
                  );
                })}
              </svg>

{interpUrl && selectedPathPoints.length >= 2 && (
                <svg
                  className="interpolation-path-svg"
                  viewBox="0 0 100 100"
                  preserveAspectRatio="none"
                >
                  <defs>
                    <marker
                      id="arrow-head"
                      markerWidth="6"
                      markerHeight="6"
                      refX="5"
                      refY="3"
                      orient="auto"
                    >
                      <path d="M0,0 L6,3 L0,6 Z" className="arrow-head" />
                    </marker>
                  </defs>

                  <polyline
                    className="interpolation-path ready"
                    points={selectedPath}
                    markerEnd="url(#arrow-head)"
                    onClick={() => {
                      interpPlayer.seek(0);
                      interpPlayer.play(interpUrl);
                    }}
                  />
                </svg>
              )}

              {placedPoints.map((point) => {
                const { color, glow } = getSoundColor(point.name, point.filename);
                return (
                  <div
                    key={point.id}
                    className={`dot${explorerSelected?.id === point.id ? " selected" : ""}${explorerSelected?.id === point.id && explorerPlayer.isPlaying ? " playing" : ""}`}
                    style={{ left: `${point.px}%`, top: `${point.py}%`, "--dot-color": color, "--dot-glow": glow } as React.CSSProperties}
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
                    <span className="dot-marker" />
                    <span className="dot-label">{point.name}</span>
                  </div>
                );
              })}
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
            <div className="timeline-zoom-controls">
              <button
                className="zoom-btn"
                onClick={() => {
                  const scroll = scrollRef.current;
                  if (scroll) zoomCenterSecRef.current = (scroll.scrollLeft + scroll.clientWidth / 2) / pxPerSec;
                  setPxPerSec((p) => Math.max(MIN_PX_PER_SEC, p / 2));
                }}
                disabled={pxPerSec <= MIN_PX_PER_SEC}
                title="Zoom out"
              >−</button>
              <span className="zoom-label">{Math.round(pxPerSec / DEFAULT_PX_PER_SEC * 100)}%</span>
              <button
                className="zoom-btn"
                onClick={() => {
                  const scroll = scrollRef.current;
                  if (scroll) zoomCenterSecRef.current = (scroll.scrollLeft + scroll.clientWidth / 2) / pxPerSec;
                  setPxPerSec((p) => Math.min(MAX_PX_PER_SEC, p * 2));
                }}
                disabled={pxPerSec >= MAX_PX_PER_SEC}
                title="Zoom in"
              >+</button>
            </div>
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
                <a
                  href={interpUrl}
                  download={`interpolation-${Date.now()}.wav`}
                  className="download-btn"
                >
                  Download WAV
                </a>
              </div>
            )}
          </div>
          <div className="timeline-header-right">
            <button
              className={`interpolate-btn${interpLoading ? " loading" : ""}`}
              onClick={runInterpolation}
              disabled={!canInterpolate || interpLoading}
            >
              {interpLoading ? (
                <>
                  {LOADING_VERBS[loadingVerbIdx]}
                  <span className="loading-dots" aria-hidden="true" />
                </>
              ) : (
                "Interpolate"
              )}
            </button>
          </div>
        </div>

        <div className="timeline-scroll" ref={scrollRef}>
          <div className="timeline-ruler" style={{ width: `${timelineWidth}px` }}>
            {Array.from({ length: rulerMarkCount + 1 }, (_, i) => (
              <div key={i} className="timeline-ruler-mark" style={{ left: `${i * rulerInterval * pxPerSec}px` }}>
                <div className="timeline-ruler-tick" />
                <span className="timeline-ruler-label">{i * rulerInterval}s</span>
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
            onClick={(e) => {
              if (e.target === e.currentTarget) setSelectedClipId(null);
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
                start: Math.max(0, dropX / pxPerSec - DEFAULT_CLIP_DURATION_SEC * (DEFAULT_PX_PER_SEC / pxPerSec) / 2),
                duration: DEFAULT_CLIP_DURATION_SEC * (DEFAULT_PX_PER_SEC / pxPerSec),
              };
              setTimelineClips((prev) => [...prev, newClip]);
              setInterpUrl("");
            }}
          >
            {timelineClips.length === 0 && (
              <div className="timeline-empty-hint">Drag sounds here to build the timeline</div>
            )}

            {snapJoints.map((x) => (
              <div key={x} className="snap-joint" style={{ left: `${x}px` }} />
            ))}

            {overlapRegions.map((region, i) => (
              <div
                key={i}
                className="timeline-overlap-region"
                style={{
                  left: `${region.start * pxPerSec}px`,
                  width: `${(region.end - region.start) * pxPerSec}px`,
                }}
              />
            ))}

            {gapRegions.map((region) => {
              const isInterpolated = interpolatedGaps.has(region.key);
              return (
                <div
                  key={region.key}
                  className={`timeline-gap-region${isInterpolated ? " interpolated" : ""}`}
                  style={{
                    left: `${region.start * pxPerSec}px`,
                    width: `${(region.end - region.start) * pxPerSec}px`,
                  }}
                  onClick={(e) => {
                    e.stopPropagation();
                    if (isInterpolated) return;
                    setInterpolatedGaps((prev) => new Set(prev).add(region.key));
                  }}
                  title={isInterpolated ? undefined : "Click to add interpolation"}
                >
                  {!isInterpolated && <span className="timeline-gap-label">+</span>}
                  {isInterpolated && (
                    <button
                      className="delete-clip-btn"
                      onClick={(e) => {
                        e.stopPropagation();
                        setInterpolatedGaps((prev) => { const s = new Set(prev); s.delete(region.key); return s; });
                      }}
                      title="Remove interpolation"
                    >✕</button>
                  )}
                </div>
              );
            })}

            {timelineClips.map((clip) => {
              const isSelected = selectedClipId === clip.id;
              const leftPx = clip.start * pxPerSec;
              const widthPx = clip.duration * pxPerSec;
              return (
                <div
                  key={clip.id}
                  className={`timeline-clip${draggingId === clip.id ? " dragging" : ""}${isSelected ? " selected" : ""}`}
                  draggable={!resizing}
                  onDragStart={(e) => {
                    if (resizing) { e.preventDefault(); return; }
                    setDraggingId(clip.id);
                    setDragStartWidth(timelineWidth);
                    setSelectedClipId(null);
                  }}
                  onDragEnd={() => { setDraggingId(null); resetDragState(); }}
                  onDrag={(e) => {
                    if (e.clientX <= 0) return;
                    const timelineLeft = e.currentTarget.parentElement?.getBoundingClientRect().left ?? 0;
                    const rawStart = (e.clientX - timelineLeft) / pxPerSec - clip.duration / 2;
                    const targets = clipEdgeTargets(clipsRef.current, clip.id);
                    const snapThreshold = SNAP_THRESHOLD_PX / pxPerSec;
                    const left = snapEdge(rawStart, targets, snapThreshold);
                    const right = snapEdge(rawStart + clip.duration, targets, snapThreshold);
                    const snapped = left.dist <= right.dist ? left.snapped : right.snapped - clip.duration;
                    moveClip(clip.id, Math.max(0, snapped));
                  }}
                  onClick={(e) => {
                    e.stopPropagation();
                    setSelectedClipId((prev) => (prev === clip.id ? null : clip.id));
                  }}
                  style={{ left: `${leftPx}px`, width: `${widthPx}px` }}
                >
                  {isSelected && (
                    <div
                      className="clip-resize-handle clip-resize-left"
                      onMouseDown={(e) => {
                        e.stopPropagation();
                        e.preventDefault();
                        setResizing({ id: clip.id, edge: "left", startMouseX: e.clientX, startSec: clip.start, startDur: clip.duration });
                      }}
                    />
                  )}

                <div className="clip-label">
                  <span className="clip-emoji">
                    {getEmoji(clip.name, clip.filename)}
                  </span>

                  <span>{clip.name}</span>
                </div>
                  {isSelected && (
                    <span className="clip-duration-badge">{clip.duration.toFixed(2)}s</span>
                  )}

                  {isSelected && (
                    <div
                      className="clip-resize-handle clip-resize-right"
                      onMouseDown={(e) => {
                        e.stopPropagation();
                        e.preventDefault();
                        setResizing({ id: clip.id, edge: "right", startMouseX: e.clientX, startSec: clip.start, startDur: clip.duration });
                      }}
                    />
                  )}

                  <button
                    className="delete-clip-btn"
                    onClick={(e) => { e.stopPropagation(); deleteClip(clip.id); }}
                    title="Remove"
                  >✕</button>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
