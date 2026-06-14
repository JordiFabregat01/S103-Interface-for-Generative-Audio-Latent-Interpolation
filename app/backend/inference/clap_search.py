"""Text-to-sound (and reusable) CLAP-based search over the library.

Re-uses the already-cached, L2-normalized audio embeddings written by
``embeddings.py`` (`clap_embeddings.npz`). Cosine similarity reduces to a
single matmul because both the cached audio embeddings and the freshly
computed text embedding are unit-norm.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from inference.embeddings import (
    EMBEDDINGS_CACHE,
    SoundPoint,
    _get_clap,
    _sound_key,
    get_sound_layout,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchHit:
    point: SoundPoint
    score: float


@lru_cache(maxsize=1)
def _load_cache() -> Tuple[Tuple[str, ...], np.ndarray]:
    """Return ``(filenames, embeddings)`` loaded once from disk.

    ``get_sound_layout()`` is called first so that on a cold start (no cache on
    disk yet) the embeddings + layout are computed and persisted before we try
    to read them.
    """
    get_sound_layout()
    if not EMBEDDINGS_CACHE.exists():
        raise FileNotFoundError(
            f"CLAP embeddings cache missing at {EMBEDDINGS_CACHE}"
        )
    data = np.load(EMBEDDINGS_CACHE, allow_pickle=True)
    filenames = tuple(str(n) for n in data["filenames"])
    embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    logger.info(
        "loaded CLAP search cache: %d files, embedding dim=%d",
        len(filenames),
        embeddings.shape[1] if embeddings.ndim == 2 else -1,
    )
    return filenames, embeddings


def invalidate_cache() -> None:
    """Drop the in-memory cache so the next search reloads from disk.

    Call this after writing a new ``clap_embeddings.npz`` (e.g. after a future
    upload endpoint extends the library).
    """
    _load_cache.cache_clear()


def _embed_text(query: str) -> np.ndarray:
    """Return a unit-norm CLAP text embedding as a ``(1024,)`` float32 array."""
    clap = _get_clap()
    with torch.no_grad():
        emb = clap.get_text_embeddings([query])
        emb = F.normalize(emb, p=2, dim=1)
    return emb[0].detach().cpu().float().numpy()


def _rank_by_vector(
    query_vec: np.ndarray,
    k: int,
    *,
    exclude_idx: int | None = None,
) -> List[SearchHit]:
    """Cosine-rank the cached library against ``query_vec``.

    ``query_vec`` must be a unit-norm ``(1024,)`` float array. ``exclude_idx``,
    when set, masks that row's score to ``-inf`` so the corresponding sound
    can't end up in the top-k (used to drop the self-hit in "more like this"
    queries).
    """
    if k <= 0:
        return []

    filenames, embeddings = _load_cache()
    if embeddings.size == 0:
        return []

    # (N,) cosine since both sides unit-norm. `asarray` pins the result to a
    # concrete float dtype so type-checkers don't widen it into a union that
    # includes bool (which would reject the unary `-` below).
    scores = np.asarray(embeddings @ query_vec, dtype=np.float32)
    if exclude_idx is not None and 0 <= exclude_idx < len(scores):
        scores[exclude_idx] = -np.inf

    # Cap k to the number of candidates that aren't masked out.
    candidates = len(scores) - (1 if exclude_idx is not None else 0)
    k = min(k, candidates)
    if k <= 0:
        return []

    # argpartition gives the top-k unsorted in O(N); a second argsort orders
    # only those k entries. Overkill for 25 files but trivial and scales.
    neg_scores = -scores
    top_idx = np.argpartition(neg_scores, k - 1)[:k]
    top_idx = top_idx[np.argsort(neg_scores[top_idx])]

    layout = {_sound_key(p.kind, p.filename): p for p in get_sound_layout()}
    hits: List[SearchHit] = []
    for idx in top_idx:
        key = filenames[idx]
        point = layout.get(key)
        if point is None:
            logger.warning(
                "search cache references %r but layout has no matching point; "
                "skipping (caches may be out of sync)",
                key,
            )
            continue
        hits.append(SearchHit(point=point, score=float(scores[idx])))
    return hits


def search_by_text(query: str, k: int = 8) -> List[SearchHit]:
    """Rank library sounds by cosine similarity to ``query`` in CLAP space.

    Returns at most ``k`` hits, sorted by descending score. An empty / blank
    query, or ``k <= 0``, yields an empty list.
    """
    query = (query or "").strip()
    if not query or k <= 0:
        return []

    text_vec = _embed_text(query)
    return _rank_by_vector(text_vec, k)


def search_similar(filename: str, kind: str, k: int = 8) -> List[SearchHit]:
    """Rank library sounds by cosine similarity to the ``(kind, filename)`` sound.

    The query sound itself is always excluded from the result, so a library
    of ``N`` sounds can return at most ``N - 1`` neighbours.

    Raises :class:`KeyError` if the sound isn't present in the cached embedding
    matrix (typically because it was added after the last cache build).
    """
    filename = (filename or "").strip()
    if not filename or k <= 0:
        return []

    key = _sound_key(kind, filename)
    filenames, embeddings = _load_cache()
    try:
        idx = filenames.index(key)
    except ValueError as exc:
        raise KeyError(
            f"{key!r} not present in CLAP embeddings cache"
        ) from exc

    query_vec = np.asarray(embeddings[idx], dtype=np.float32)
    return _rank_by_vector(query_vec, k, exclude_idx=idx)
