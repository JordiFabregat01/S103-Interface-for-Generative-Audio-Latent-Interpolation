import { useEffect, useMemo, useRef, useState } from "react";
import "../App.css";
import { getSounds, getSoundUrl, searchSounds, findSimilarSounds, render, cancelRender, type Segment, type SoundPoint, type SoundHit, type Kind } from "../api";
import { useAudioPlayer } from "../hooks/useAudioPlayer";

// start and duration are stored in seconds; pixels = value * pxPerSec
type TimelineClip = {
  id: number;
  name: string;
  filename: string;
  kind: Kind;
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
const STORAGE_KEY = "gali-workspace";
const LOADING_VERBS = ["crunching", "baking", "brewing", "simmering", "blending", "cooking", "hallucinating", "conjuring", "morphing", "weaving", "vibing", "crafting"];

const MIN_ZOOM = 0.5;
const MAX_ZOOM = 3;
const PATH_SNAP_RADIUS = 4; // latent units (0–100 viewBox)

type ExplorerCategory = { key: string; label: string; emoji: string; color: string; match: (lower: string) => boolean };

const CATEGORY_DEFS: ExplorerCategory[] = [
  { key: "wind", label: "Wind", emoji: "💨", color: "#6ee7d4", match: (l) => l.includes("wind") || l.includes("breeze") },
  { key: "storm", label: "Storm", emoji: "⛈️", color: "#a78bfa", match: (l) => l.includes("thunder") || l.includes("storm") },
  { key: "rain", label: "Rain", emoji: "🌧️", color: "#7dd3fc", match: (l) => l.includes("rain") },
  { key: "water", label: "Water", emoji: "💧", color: "#38bdf8", match: (l) => l.includes("waterfall") || l.includes("waterrocks") || l.includes("underwater") || l.includes("river") || l.includes("sea") || l.includes("waves") || l.includes("water") },
  { key: "fire", label: "Fire", emoji: "🔥", color: "#fb923c", match: (l) => l.includes("fire") },
  { key: "birds", label: "Birds", emoji: "🐦", color: "#86efac", match: (l) => l.includes("bird") || l.includes("seagull") || l.includes("loon") },
  { key: "insects", label: "Insects", emoji: "🦗", color: "#fde047", match: (l) => l.includes("bee") || l.includes("cicada") || l.includes("cricket") },
  { key: "human", label: "Human", emoji: "🥾", color: "#c084fc", match: (l) => l.includes("footstep") || l.includes("keyboard") || l.includes("step") },
];

function getCategoryKey(name: string, filename = ""): string {
  const lower = `${name} ${filename}`.toLowerCase();
  for (const c of CATEGORY_DEFS) if (c.match(lower)) return c.key;
  return "other";
}

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
  const [pathPreviewing, setPathPreviewing] = useState(false);
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
  //const STORAGE_KEY = "gali-workspace";
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

  // Explorer pan/zoom state
  const [panX, setPanX] = useState(0);
  const [panY, setPanY] = useState(0);
  const [zoom, setZoom] = useState(1);
  const panRef = useRef({ x: 0, y: 0, zoom: 1 });
  useEffect(() => { panRef.current = { x: panX, y: panY, zoom }; }, [panX, panY, zoom]);
  const panDragRef = useRef<{ startX: number; startY: number; origPanX: number; origPanY: number } | null>(null);
  const explorerPlotRef = useRef<HTMLDivElement>(null);

  // Path tool state
  const [pathMode, setPathMode] = useState(false);
  const pathModeRef = useRef(false);
  useEffect(() => { pathModeRef.current = pathMode; }, [pathMode]);
  const [isDrawingPath, setIsDrawingPath] = useState(false);
  const isDrawingPathRef = useRef(false);
  useEffect(() => { isDrawingPathRef.current = isDrawingPath; }, [isDrawingPath]);
  const [drawnPath, setDrawnPath] = useState<{ x: number; y: number }[]>([]);
  const [drawnNodes, setDrawnNodes] = useState<SoundPoint[]>([]);
  const drawnNodesRef = useRef<SoundPoint[]>([]);
  useEffect(() => { drawnNodesRef.current = drawnNodes; }, [drawnNodes]);
  const [hasDrawnPath, setHasDrawnPath] = useState(false);
  const hasDrawnPathRef = useRef(false);
  useEffect(() => { hasDrawnPathRef.current = hasDrawnPath; }, [hasDrawnPath]);

  // Filters
  const [hiddenCategories, setHiddenCategories] = useState<Set<string>>(new Set());

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

  const [libraryTab, setLibraryTab] = useState<Kind>("long");
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

  const [renderStatus, setRenderStatus] = useState<"queued" | "running" | null>(null);
  const [renderProgress, setRenderProgress] = useState<{ done: number; total: number } | null>(null);
  const [renderJobId, setRenderJobId] = useState<string | null>(null);
  const interpLoading = renderStatus !== null;
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
      setLoadingVerbIdx(Math.floor(Math.random() * LOADING_VERBS.length));
    }, 1400 * 2);
    return () => clearInterval(id);
  }, [interpLoading]);

  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY);

    if (!saved) return;

    try {
      const parsed = JSON.parse(saved);
      console.log("LOADED", parsed);


    if (parsed.timelineClips) {
      setTimelineClips(parsed.timelineClips);
    }

    if (parsed.quality) {
      setQuality(parsed.quality);
    }
   if (parsed.interpolatedGaps) {
    setInterpolatedGaps(new Set(parsed.interpolatedGaps));
  }

  } catch (err) {
    console.error("Failed to load workspace:", err);
  }
}, []);

 useEffect(() => {
  if (timelineClips.length === 0) return;

  const data = {
    timelineClips,
    quality,
    interpolatedGaps: [...interpolatedGaps],

  };

  console.log("Saving workspace");

  localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
}, [timelineClips, quality, interpolatedGaps]);

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
  if (lower.includes("snow")) return "🥾❄️";
  if (lower.includes("footstep")) return "🦶";
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
      segments.push({ type: "clip", filename: clip.filename, kind: clip.kind, duration: clip.duration });
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
            audio1_kind: clip.kind,
            audio2_kind: next.kind,
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
          audio1_kind: clip.kind,
          audio2_kind: next.kind,
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
    if (sorted.length < 1) return;

    setRenderStatus("queued");
    setRenderProgress(null);
    setRenderJobId(null);
    setInterpError(null);
    if (interpUrlRef.current) {
      URL.revokeObjectURL(interpUrlRef.current);
      interpUrlRef.current = null;
    }
    setInterpUrl(null);
    try {
      const url = await render(buildTimelineSegments(sorted), {
        onJobId: (id) => setRenderJobId(id),
        onProgress: (snap) => {
          if (snap.status === "queued" || snap.status === "running") {
            setRenderStatus(snap.status);
            if (snap.progress) setRenderProgress(snap.progress);
          }
        },
      });
      interpUrlRef.current = url;
      setInterpUrl(url);
    } catch (err) {
      setInterpError(err instanceof Error ? err.message : "Render failed");
    } finally {
      setRenderStatus(null);
      setRenderProgress(null);
      setRenderJobId(null);
    }
  };

  const cancelRunningRender = async () => {
    if (!renderJobId) return;
    try {
      await cancelRender(renderJobId);
    } catch {
      /* polling loop will surface the resulting error */
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

  const canInterpolate = timelineClips.length >= 1;

  const placedPoints = useMemo(
    () => sounds.filter((p) => p.kind === libraryTab).map((p) => ({ ...p, px: p.x * 100, py: p.y * 100 })),
    [sounds, libraryTab],
  );

const selectedPathPoints = sortedClips
  .map((clip) =>
    placedPoints.find((point) => point.filename === clip.filename)
  )
  .filter(Boolean);

const selectedPath = selectedPathPoints
  .map((point) => `${point!.px},${point!.py-2}`)
  .join(" ");

  const visiblePoints = useMemo(
    () => placedPoints.filter((p) => !hiddenCategories.has(getCategoryKey(p.name, p.filename))),
    [placedPoints, hiddenCategories]
  );
  const visiblePointsRef = useRef(visiblePoints);
  useEffect(() => { visiblePointsRef.current = visiblePoints; }, [visiblePoints]);

  const transformAttr = `translate(${panX} ${panY}) scale(${zoom})`;

  const zoomBy = (factor: number, cx = 50, cy = 50) => {
    const z = panRef.current.zoom;
    const next = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, z * factor));
    if (next === z) return;
    const latX = (cx - panRef.current.x) / z;
    const latY = (cy - panRef.current.y) / z;
    setZoom(next);
    setPanX(cx - latX * next);
    setPanY(cy - latY * next);
  };
  const goHome = () => { setPanX(0); setPanY(0); setZoom(1); };

  const nearestDot = (latX: number, latY: number): SoundPoint | null => {
    let best: SoundPoint | null = null;
    let bestDist = PATH_SNAP_RADIUS;
    for (const p of visiblePointsRef.current) {
      const d = Math.hypot(p.x * 100 - latX, p.y * 100 - latY);
      if (d < bestDist) { bestDist = d; best = p; }
    }
    return best;
  };

  // Global mouse handlers for panning and path drawing
  useEffect(() => {
    const screenToLatent = (clientX: number, clientY: number) => {
      const el = explorerPlotRef.current;
      if (!el) return null;
      const rect = el.getBoundingClientRect();
      const sx = ((clientX - rect.left) / rect.width) * 100;
      const sy = ((clientY - rect.top) / rect.height) * 100;
      const { x: px, y: py, zoom: z } = panRef.current;
      return { lx: (sx - px) / z, ly: (sy - py) / z };
    };
    const onMove = (e: MouseEvent) => {
      if (panDragRef.current) {
        const el = explorerPlotRef.current;
        if (!el) return;
        const rect = el.getBoundingClientRect();
        const dx = ((e.clientX - panDragRef.current.startX) / rect.width) * 100;
        const dy = ((e.clientY - panDragRef.current.startY) / rect.height) * 100;
        setPanX(panDragRef.current.origPanX + dx);
        setPanY(panDragRef.current.origPanY + dy);
      } else if (isDrawingPathRef.current) {
        const pt = screenToLatent(e.clientX, e.clientY);
        if (!pt) return;
        setDrawnPath((prev) => [...prev, { x: pt.lx, y: pt.ly }]);
        const hit = nearestDot(pt.lx, pt.ly);
        if (hit) {
          const cur = drawnNodesRef.current;
          if (cur.length === 0 || cur[cur.length - 1].id !== hit.id) {
            setDrawnNodes([...cur, hit]);
          }
        }
      }
    };
    const onUp = () => {
      if (panDragRef.current) {
        panDragRef.current = null;
        document.body.style.cursor = "";
      }
      if (isDrawingPathRef.current) {
        setIsDrawingPath(false);
        const nodes = drawnNodesRef.current;
        if (nodes.length >= 2) {
          setHasDrawnPath(true);
        } else {
          setDrawnPath([]);
          setDrawnNodes([]);
        }
      }
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
    return () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
  }, []);

  // Wheel zoom on explorer plot
  useEffect(() => {
    const el = explorerPlotRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      const cx = ((e.clientX - rect.left) / rect.width) * 100;
      const cy = ((e.clientY - rect.top) / rect.height) * 100;
      zoomBy(e.deltaY < 0 ? 1.2 : 1 / 1.2, cx, cy);
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  const onPlotMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    if (pathModeRef.current) {
      const el = explorerPlotRef.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      const sx = ((e.clientX - rect.left) / rect.width) * 100;
      const sy = ((e.clientY - rect.top) / rect.height) * 100;
      const { x: px, y: py, zoom: z } = panRef.current;
      const lx = (sx - px) / z;
      const ly = (sy - py) / z;
      setIsDrawingPath(true);
      setDrawnPath([{ x: lx, y: ly }]);
      const hit = nearestDot(lx, ly);
      setDrawnNodes(hit ? [hit] : []);
    } else {
      panDragRef.current = {
        startX: e.clientX,
        startY: e.clientY,
        origPanX: panRef.current.x,
        origPanY: panRef.current.y,
      };
      document.body.style.cursor = "grabbing";
    }
  };

  const cancelPath = () => {
    setHasDrawnPath(false);
    setDrawnPath([]);
    setDrawnNodes([]);
  };

  const acceptPath = () => {
    if (drawnNodes.length < 2) return;
    const startSec = timelineClips.reduce((max, c) => Math.max(max, c.start + c.duration), 0);
    const dur = DEFAULT_CLIP_DURATION_SEC;
    const t = Date.now();
    const newClips: TimelineClip[] = drawnNodes.map((p, i) => ({
      id: t + i,
      name: p.name,
      filename: p.filename,
      kind: p.kind,
      start: startSec + i * dur,
      duration: dur,
    }));
    setTimelineClips((prev) => [...prev, ...newClips]);
    setInterpUrl("");
    setPathMode(false);
    cancelPath();
  };

  const loadExampleTimeline = () => {
    if (sounds.length < 2) return;

    const pick = (keywords: string[]) =>
      sounds.find(s =>
        keywords.some(k => (s.name + " " + s.filename).toLowerCase().includes(k))
      );

    const a = pick(["camp_fire", "campfire", "fire"]) ?? sounds[0];
    const b = pick(["keyboard"]) ?? sounds.find(s => s.filename !== a.filename) ?? sounds[1];

    const t = Date.now();
    setTimelineClips([
      { id: t,     name: a.name, filename: a.filename, kind: a.kind, start: 0, duration: 5 },
      { id: t + 1, name: b.name, filename: b.filename, kind: b.kind, start: 9, duration: 5 },
    ]);
    setInterpUrl("");
    setShowHowToUse(false);
  };

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

              <hr className="how-to-divider" />

              <div className="how-to-demo">
                <p className="demo-audio-label">Try it — campfire · gap · keyboard</p>

                <img
                  src="/example-timeline.png"
                  alt="Example timeline: campfire, gap, keyboard"
                  className="demo-timeline-img"
                />

                <div className="how-to-demo-media">
                  <img
                    src="/demo-drag.gif"
                    alt="Drag a sound card onto the timeline"
                    className="demo-gif"
                    onError={(e) => {
                      const target = e.currentTarget;
                      target.style.display = "none";
                      const placeholder = target.nextElementSibling as HTMLElement | null;
                      if (placeholder) placeholder.style.display = "flex";
                    }}
                  />
                  <div className="demo-gif-placeholder" style={{ display: "none" }}>
                    demo-drag.gif · drop into public/ to show here
                  </div>
                </div>

                <div className="demo-audio-row">
                  <audio controls src="/example-output.wav" className="demo-audio-player" />
                  <button
                    className="how-to-demo-cta"
                    onClick={loadExampleTimeline}
                    disabled={sounds.length < 2}
                  >
                    Load Example →
                  </button>
                </div>
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

              <p className="about-tagline">
                <span style={{ color: "#1E90FF" }}>GALI</span>
                {" "}— Generative Audio Latent Interpolation
              </p>

              <p className="about-subtitle">
                Explore ambient soundscapes through AI-powered latent space navigation.
              </p>

              <div className="about-section">
                <h4>Tech Stack</h4>

                <div className="about-tech-grid">

                  <div>
                    <p className="about-tech-title">Frontend</p>
                    <ul className="about-feature-list">
                      <li>React</li>
                      <li>TypeScript</li>
                      <li>Vite</li>
                      <li>Web Audio API</li>
                    </ul>
                  </div>

                  <div>
                    <p className="about-tech-title">Backend</p>
                    <ul className="about-feature-list">
                      <li>Python</li>
                      <li>Flask</li>
                      <li>Audio generation pipeline</li>
                    </ul>
                  </div>

                  <div>
                    <p className="about-tech-title">AI / Audio</p>
                    <ul className="about-feature-list">
                      <li>CLAP embeddings</li>
                      <li>Latent-space interpolation</li>
                    </ul>
                  </div>

                </div>
              </div>
              <div className="about-divider" />

              <div className="about-section">
                <h4>What is GALI?</h4>

                <p>
                  GALI is an experimental browser-based interface for exploring,
                  blending and interpolating environmental audio using generative AI.
                </p>

                <p>
                  Sounds are embedded into a shared latent space using the
                  <strong> CLAP </strong>
                  audio-language model, enabling semantic search,
                  similarity discovery and seamless interpolation between soundscapes.
                </p>
              </div>

              <div className="about-section">
                <h4>Core Features</h4>

                <ul className="about-feature-list">
                  <li>AI-based sound similarity search</li>
                  <li>Interactive latent space exploration</li>
                  <li>Timeline-based audio composition</li>
                  <li>Crossfades & generative interpolations</li>
                  <li>Browser-native audio rendering workflow</li>
                  <li>Local workspace auto-save</li>
                </ul>
              </div>

              <div className="about-section">
                <h4>Workflow</h4>

                <p>
                  Drag sounds into the timeline, create transitions between clips,
                  explore neighbouring sounds in latent space, and render fully blended
                  ambient compositions directly in the browser.
                </p>
              </div>

              <div className="about-divider" />

              <p className="about-credits">
                Built as part of the Music Technology Group at
                <strong> Universitat Pompeu Fabra</strong>
              </p>

            </div>
          </div>
        </>
      )}

      <div className="workspace-layout">
        <div className="library-panel">
          <h2>Sound Library</h2>
          <p className="library-hint">Click to preview · Drag to timeline</p>

          <div className="library-tabs">
            {(["long", "short"] as Kind[]).map((k) => (
              <button
                key={k}
                className={`library-tab${libraryTab === k ? " active" : ""}`}
                onClick={() => { setLibraryTab(k); setSelectedSound(null); previewPlayer.pause(); }}
              >
                {k === "long" ? "Long" : "Short"}
              </button>
            ))}
          </div>

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
                const inTab = (s: SoundPoint) => s.kind === libraryTab;
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
                        previewPlayer.play(getSoundUrl(sound.filename, sound.kind));
                      }
                    }}
                    onContextMenu={(e) => { e.preventDefault(); setContextMenu({ x: e.clientX, y: e.clientY, sound }); }}
                    onDragStart={() => { dragSoundRef.current = sound; }}
                    onDragEnd={() => { dragSoundRef.current = null; resetDragState(); }}
                  >
                    <div className="sound-image">{getEmoji(sound.name, sound.filename)}</div>
                    <p>{sound.name}</p>
                  </div>
                );

                if (similarTo && searchResults !== null) {
                  const similarIds = new Set(searchResults.map((s) => s.id));
                  const others = sounds.filter((s) => inTab(s) && !similarIds.has(s.id) && s.id !== similarTo.id);
                  return (
                    <>
                      {searchResults.filter(inTab).map((s) => renderCard(s, "similar"))}
                      {inTab(similarTo) && renderCard(similarTo, "similar-origin")}
                      {others.map((s) => renderCard(s))}
                    </>
                  );
                }

                return (searchResults ?? sounds).filter(inTab).map((s) => renderCard(s));
              })()}
            </div>
          </div>

          {selectedSound && (
            <div className="sound-preview-panel">
              <div className="sound-preview-header">
                <span className="preview-name">{getEmoji(selectedSound.name, selectedSound.filename)} {selectedSound.name}</span>
                <button className="close-preview-btn" onClick={() => { previewPlayer.pause(); setSelectedSound(null); }}>✕</button>
              </div>
              <div className="preview-controls">
                <button
                  className="preview-play-btn"
                  onClick={() => { interpPlayer.pause(); previewPlayer.seek(0); previewPlayer.play(getSoundUrl(selectedSound.filename, selectedSound.kind)); }}
                  title="Restart"
                >↺</button>
                <button
                  className="preview-play-btn"
                  onClick={() => previewPlayer.isPlaying ? previewPlayer.pause() : (interpPlayer.pause(), previewPlayer.play(getSoundUrl(selectedSound.filename, selectedSound.kind)))}
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
                  findSimilarSounds(selectedSound.filename, selectedSound.kind)
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
                findSimilarSounds(contextMenu.sound.filename, contextMenu.sound.kind)
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
          <p className="library-hint">
            {pathMode
              ? "Drag through sounds to trace a path · release to confirm"
              : "Click to preview · Drag to timeline · Drag empty space to pan"}
          </p>
          <div className="explorer-body">
            <div className={`explorer-plot-wrap${pathMode ? " path-mode" : ""}${panDragRef.current ? " panning" : ""}`}>
              <div
                className="explorer-plot"
                ref={explorerPlotRef}
                onMouseDown={onPlotMouseDown}
              >
                {/* Drawn path (live or pending review) */}
                {(isDrawingPath || drawnPath.length > 0) && (
                  <svg className="drawn-path-svg" viewBox="0 0 100 100" preserveAspectRatio="none">
                    <g transform={transformAttr}>
                      <polyline
                        className="drawn-path-line"
                        points={drawnPath.map((p) => `${p.x},${p.y}`).join(" ")}
                      />
                      {drawnNodes.map((n) => (
                        <circle key={n.id} className="drawn-path-node" cx={n.x * 100} cy={n.y * 100} r="2.4" />
                      ))}
                    </g>
                  </svg>
                )}

                {selectedPathPoints.length >= 2 && (
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
                    <g transform={transformAttr}>
                      <polyline
                        className="interpolation-path ready"
                        points={selectedPath}
                        markerEnd="url(#arrow-head)"
                        onClick={() => {
                          if (!interpUrl) return;
                          explorerPlayer.pause();
                          previewPlayer.pause();
                          setExplorerSelected(null);
                          setPathPreviewing(true);
                          interpPlayer.seek(0);
                          interpPlayer.play(interpUrl);
                        }}
                      />
                    </g>
                  </svg>
                )}

                {visiblePoints.map((point) => {
                  const { color, glow } = getSoundColor(point.name, point.filename);
                  const dx = point.px * zoom + panX;
                  const dy = point.py * zoom + panY;
                  if (dx < -5 || dx > 105 || dy < -5 || dy > 105) return null;
                  return (
                    <div
                      key={point.id}
                      className={`dot${explorerSelected?.id === point.id ? " selected" : ""}${explorerSelected?.id === point.id && explorerPlayer.isPlaying ? " playing" : ""}`}
                      style={{ left: `${dx}%`, top: `${dy}%`, "--dot-color": color, "--dot-glow": glow } as React.CSSProperties}
                      draggable={!pathMode}
                      onMouseDown={(e) => { e.stopPropagation(); }}
                      onDragStart={() => { dragSoundRef.current = point; }}
                      onDragEnd={() => { dragSoundRef.current = null; resetDragState(); }}
                      onClick={(e) => {
                        e.stopPropagation();
                        if (pathMode) return;
                        if (explorerSelected?.id === point.id) {
                          if (explorerPlayer.isPlaying) { explorerPlayer.pause(); } else { explorerPlayer.play(getSoundUrl(point.filename, point.kind)); }
                        } else {
                          explorerPlayer.pause();
                          interpPlayer.pause();
                          setPathPreviewing(false);
                          setExplorerSelected(point);
                          explorerPlayer.play(getSoundUrl(point.filename, point.kind));
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

              {explorerSelected && (
                <div className="sound-preview-panel explorer-floating-preview">
                  <div className="sound-preview-header">
                    <span className="preview-name">{getEmoji(explorerSelected.name, explorerSelected.filename)} {explorerSelected.name}</span>
                    <button className="close-preview-btn" onClick={() => { explorerPlayer.pause(); setExplorerSelected(null); }}>✕</button>
                  </div>
                  <div className="preview-controls">
                    <button className="preview-play-btn" onClick={() => { explorerPlayer.seek(0); explorerPlayer.play(getSoundUrl(explorerSelected.filename, explorerSelected.kind)); }} title="Restart">↺</button>
                    <button className="preview-play-btn" onClick={() => explorerPlayer.isPlaying ? explorerPlayer.pause() : explorerPlayer.play(getSoundUrl(explorerSelected.filename, explorerSelected.kind))}>
                      {explorerPlayer.isPlaying ? "⏸" : "▶"}
                    </button>
                    <input className="audio-scrubber" type="range" min={0} max={explorerPlayer.duration || 1} step={0.01} value={explorerPlayer.currentTime} onChange={(e) => explorerPlayer.seek(Number(e.target.value))} />
                    <span className="audio-time">{fmt(explorerPlayer.currentTime)} / {fmt(explorerPlayer.duration)}</span>
                  </div>
                </div>
              )}

              {pathPreviewing && interpUrl && !explorerSelected && (
                <div className="sound-preview-panel explorer-floating-preview">
                  <div className="sound-preview-header">
                    <span className="preview-name">↝ Interpolated path</span>
                    <button className="close-preview-btn" onClick={() => { interpPlayer.pause(); setPathPreviewing(false); }}>✕</button>
                  </div>
                  <div className="preview-controls">
                    <button className="preview-play-btn" onClick={() => { interpPlayer.seek(0); interpPlayer.play(interpUrl); }} title="Restart">↺</button>
                    <button className="preview-play-btn" onClick={() => interpPlayer.isPlaying ? interpPlayer.pause() : interpPlayer.play(interpUrl)}>
                      {interpPlayer.isPlaying ? "⏸" : "▶"}
                    </button>
                    <input className="audio-scrubber" type="range" min={0} max={interpPlayer.duration || 1} step={0.01} value={interpPlayer.currentTime} onChange={(e) => interpPlayer.seek(Number(e.target.value))} />
                    <span className="audio-time">{fmt(interpPlayer.currentTime)} / {fmt(interpPlayer.duration)}</span>
                  </div>
                </div>
              )}
            </div>

            <div className="explorer-tools">
              <div className="explorer-tool-row">
                <button className="zoom-btn" onClick={() => zoomBy(1.5)} disabled={zoom >= MAX_ZOOM - 1e-6} title="Zoom in">+</button>
                <button className="zoom-btn" onClick={() => zoomBy(1 / 1.5)} disabled={zoom <= MIN_ZOOM + 1e-6} title="Zoom out">−</button>
                <button className="zoom-btn" onClick={goHome} title="Reset view">⌂</button>
              </div>
              <button
                className={`explorer-path-btn${pathMode ? " active" : ""}`}
                onClick={() => { if (pathMode) { setPathMode(false); cancelPath(); } else { setPathMode(true); } }}
                title="Trace a path through the space"
              >
                {pathMode ? "Cancel path" : "✎ Trace path"}
              </button>

              <div className="explorer-tool-divider" />

              <div className="explorer-section-label">Filters</div>
              <div className="explorer-filter-chips">
                {CATEGORY_DEFS.map((c) => {
                  const hidden = hiddenCategories.has(c.key);
                  return (
                    <button
                      key={c.key}
                      className={`filter-chip${hidden ? " off" : ""}`}
                      style={{ "--chip-color": c.color } as React.CSSProperties}
                      onClick={() => {
                        setHiddenCategories((prev) => {
                          const next = new Set(prev);
                          if (next.has(c.key)) next.delete(c.key); else next.add(c.key);
                          return next;
                        });
                      }}
                      title={`${hidden ? "Show" : "Hide"} ${c.label}`}
                    >
                      <span className="filter-chip-emoji">{c.emoji}</span>
                      <span className="filter-chip-label">{c.label}</span>
                    </button>
                  );
                })}
              </div>

              <div className="explorer-tool-divider" />

              <div className="explorer-section-label">Overview</div>
              <svg
                className="explorer-minimap"
                viewBox="-6 -6 112 112"
                preserveAspectRatio="none"
                onClick={(e) => {
                  const r = (e.currentTarget as unknown as SVGSVGElement).getBoundingClientRect();
                  // Map click into latent coords using the same padded viewBox (-6..106 → 0..100 of data).
                  const cx = (((e.clientX - r.left) / r.width) * 112) - 6;
                  const cy = (((e.clientY - r.top) / r.height) * 112) - 6;
                  setPanX(50 - cx * zoom);
                  setPanY(50 - cy * zoom);
                }}
              >
                {placedPoints.map((p) => {
                  const cat = getCategoryKey(p.name, p.filename);
                  const color = CATEGORY_DEFS.find((c) => c.key === cat)?.color ?? "#58a6ff";
                  const muted = hiddenCategories.has(cat);
                  return (
                    <circle key={p.id} cx={p.px} cy={p.py} r="1.4" fill={color} opacity={muted ? 0.18 : 0.85} />
                  );
                })}
                <rect
                  className="minimap-viewport"
                  x={-panX / zoom}
                  y={-panY / zoom}
                  width={100 / zoom}
                  height={100 / zoom}
                />
              </svg>

              {hasDrawnPath && drawnNodes.length >= 2 && (
                <>
                  <div className="explorer-tool-divider" />
                  <div className="explorer-section-label">Drawn path · {drawnNodes.length} sounds</div>
                  <div className="drawn-path-actions">
                    <button className="btn-secondary" onClick={cancelPath}>Cancel</button>
                    <button className="btn-primary" onClick={acceptPath}>Add to timeline</button>
                  </div>
                </>
              )}
            </div>
          </div>

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
            <button
              className="timeline-clear-btn"
              onClick={() => setTimelineClips([])}
              disabled={timelineClips.length === 0}
              title="Clear timeline"
            >Clear</button>
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
            <div className="render-control">
              {interpLoading && renderJobId && (
                <button
                  type="button"
                  className="render-cancel-btn"
                  onClick={cancelRunningRender}
                  title="Cancel render"
                >
                  Cancel
                </button>
              )}
              <button
                className={`interpolate-btn${interpLoading ? " loading" : ""}`}
                onClick={runInterpolation}
                disabled={!canInterpolate || interpLoading}
              >
                {renderStatus === "queued" && (
                  <>
                    Queued
                    <span className="loading-dots" aria-hidden="true" />
                  </>
                )}
                {renderStatus === "running" && (
                  <>
                    {LOADING_VERBS[loadingVerbIdx]}
                    <span className="loading-dots" aria-hidden="true" />
                  </>
                )}
                {!renderStatus && "Interpolate"}
                {renderStatus === "running" && renderProgress && renderProgress.total > 0 && (
                  <span
                    className="interpolate-btn-fill"
                    style={{
                      width: `${(renderProgress.done / Math.max(1, renderProgress.total)) * 100}%`,
                    }}
                  />
                )}
              </button>
            </div>
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
                kind: sound.kind,
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
                  onDragEnd={(e) => {
                    setDraggingId(null);
                    resetDragState();
                    const scroll = scrollRef.current;
                    if (!scroll) return;
                    const r = scroll.getBoundingClientRect();
                    const outside =
                      e.clientX < r.left || e.clientX > r.right ||
                      e.clientY < r.top || e.clientY > r.bottom;
                    if (outside) deleteClip(clip.id);
                  }}
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
      
      <div className="autosave-indicator">
        ● Auto-saved locally
      </div>
    </div>
  );
}
