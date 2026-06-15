from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List

import librosa
import numpy as np
import torch

from inference.constants import ASSETS_DIR, CACHE_DIR, SOUND_KINDS
from inference.scapes_runtime import CLAPWrapper

logger = logging.getLogger(__name__)

DATA_DIR = ASSETS_DIR
EMBEDDINGS_CACHE = CACHE_DIR / "clap_embeddings.npz"
LAYOUT_CACHE = CACHE_DIR / "tsne_layout.json"

CLAP_SR = 48000
SEGMENT_SECONDS = 8.0


@dataclass(frozen=True)
class SoundPoint:
    id: int
    name: str
    filename: str
    kind: str
    x: float
    y: float


def _sound_key(kind: str, filename: str) -> str:
    """Collision-free identity for caches/lookups, e.g. ``short/BreakingWater.wav``."""
    return f"{kind}/{filename}"


@lru_cache(maxsize=1)
def _get_clap() -> CLAPWrapper:
    use_cuda = torch.cuda.is_available()
    logger.info(f"Loading CLAPWrapper (use_cuda={use_cuda}) for embeddings...")
    return CLAPWrapper(version="2023", use_cuda=use_cuda)


def _list_wav_files() -> List[tuple[Path, str]]:
    """Return ``(path, kind)`` for every wav across the per-kind subfolders.

    Sorted by kind then name so the embedding/layout caches have a stable order.
    """
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")
    out: List[tuple[Path, str]] = []
    for kind in SOUND_KINDS:
        kind_dir = DATA_DIR / kind
        if not kind_dir.exists():
            continue
        for p in kind_dir.iterdir():
            if p.suffix.lower() == ".wav":
                out.append((p, kind))
    return sorted(out, key=lambda pk: (pk[1], pk[0].name))


def _segment_waveform(waveform: np.ndarray, sr: int, segment_sec: float = SEGMENT_SECONDS) -> List[np.ndarray]:
    seg_len = int(sr * segment_sec)
    n_segs = max(1, waveform.shape[-1] // seg_len)
    return [waveform[i * seg_len : (i + 1) * seg_len] for i in range(n_segs)]


def _compute_file_embedding(path: Path, clap: CLAPWrapper) -> np.ndarray:
    waveform, sr = librosa.load(str(path), sr=CLAP_SR, mono=True)
    segments = _segment_waveform(waveform, sr)

    seg_embeddings: List[np.ndarray] = []
    for seg in segments:
        if seg.size == 0:
            continue
        seg_tensor = torch.tensor(seg).unsqueeze(0).unsqueeze(0)
        if torch.cuda.is_available():
            seg_tensor = seg_tensor.cuda()
        emb = clap.compute_embedding(seg_tensor, og_sr=sr, random_extension=False)
        seg_embeddings.append(emb.squeeze(0).detach().cpu().numpy())

    if not seg_embeddings:
        raise ValueError(f"No usable audio segments in {path}")

    mean_emb = np.mean(np.stack(seg_embeddings), axis=0)
    norm = np.linalg.norm(mean_emb)
    return mean_emb / norm if norm > 0 else mean_emb


def _load_cached_embeddings(filenames: List[str]):
    if not EMBEDDINGS_CACHE.exists():
        return None
    try:
        cache = np.load(EMBEDDINGS_CACHE, allow_pickle=True)
        cached_names = list(cache["filenames"])
        if cached_names == filenames:
            logger.info(f"Loaded cached CLAP embeddings ({len(cached_names)} files).")
            return cache["embeddings"]
    except Exception as exc:
        logger.warning(f"Failed to read embeddings cache, recomputing: {exc}")
    return None


def _save_embeddings_cache(filenames: List[str], embeddings: np.ndarray) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(EMBEDDINGS_CACHE, filenames=np.array(filenames), embeddings=embeddings)


def _compute_all_embeddings(wav_files: List[tuple[Path, str]]) -> np.ndarray:
    keys = [_sound_key(kind, p.name) for p, kind in wav_files]
    cached = _load_cached_embeddings(keys)
    if cached is not None:
        return cached

    clap = _get_clap()
    logger.info(f"Computing CLAP embeddings for {len(wav_files)} files...")
    embeddings = np.stack([_compute_file_embedding(p, clap) for p, _ in wav_files])
    _save_embeddings_cache(keys, embeddings)
    return embeddings


def _run_tsne(embeddings: np.ndarray) -> np.ndarray:
    from sklearn.manifold import TSNE

    n_samples = embeddings.shape[0]
    perplexity = max(2, min(5, n_samples - 1))
    logger.info(f"Running t-SNE on {n_samples} points (perplexity={perplexity})...")
    reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca")
    return reducer.fit_transform(embeddings)


def _normalize(coords: np.ndarray) -> np.ndarray:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    spans = np.where(maxs - mins == 0, 1.0, maxs - mins)
    return (coords - mins) / spans


def _layout_from_files(wav_files: List[tuple[Path, str]]) -> List[SoundPoint]:
    embeddings = _compute_all_embeddings(wav_files)
    coords_2d = _run_tsne(embeddings)
    coords_norm = _normalize(coords_2d)

    return [
        SoundPoint(
            id=i,
            name=path.stem,
            filename=path.name,
            kind=kind,
            x=float(coords_norm[i, 0]),
            y=float(coords_norm[i, 1]),
        )
        for i, (path, kind) in enumerate(wav_files)
    ]


def get_sound_layout(force: bool = False) -> List[SoundPoint]:
    wav_files = _list_wav_files()
    keys = [_sound_key(kind, p.name) for p, kind in wav_files]

    if not force and LAYOUT_CACHE.exists():
        try:
            with LAYOUT_CACHE.open("r", encoding="utf-8") as f:
                cached = json.load(f)
            cached_keys = [_sound_key(item["kind"], item["filename"]) for item in cached]
            if cached_keys == keys:
                logger.info("Returning cached t-SNE layout.")
                return [SoundPoint(**item) for item in cached]
        except Exception as exc:
            logger.warning(f"Failed to read layout cache, recomputing: {exc}")

    layout = _layout_from_files(wav_files)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with LAYOUT_CACHE.open("w", encoding="utf-8") as f:
        json.dump([point.__dict__ for point in layout], f, indent=2)

    return layout


def resolve_audio_file(filename: str, kind: str) -> Path:
    safe_name = Path(filename).name
    if kind not in SOUND_KINDS:
        raise FileNotFoundError(f"Unknown sound kind: {kind!r}")
    candidate = DATA_DIR / kind / safe_name
    if not candidate.exists() or candidate.suffix.lower() != ".wav":
        raise FileNotFoundError(f"Audio file not found: {kind}/{safe_name}")
    return candidate
