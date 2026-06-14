"""Canonical encoded-source cache for the backend.

Replaces the bare-stem `.pt` files SCAPES used to write next to the audio
(or rather, next to the CWD), with a single canonical directory under
`assets/cache/`. One file per source: `{source_id}.source.pt`.

`get_or_encode` is the single entry point the rest of the backend uses.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import torch

from inference.constants import CACHE_DIR, audio_asset_path, source_cache_key
from inference.interpolation import EncodedSource
from inference.scapes_runtime import FlowInference

logger = logging.getLogger(__name__)


SOURCE_CACHE_VERSION = 1


def source_cache_path(source_id: str) -> Path:
    return CACHE_DIR / f"{source_id}.source.pt"


def save_encoded_source(source: EncodedSource) -> Path:
    """Atomically save an EncodedSource to the canonical cache directory."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = source_cache_path(source.source_id)

    payload = {
        "version": SOURCE_CACHE_VERSION,
        "source_id": source.source_id,
        "sample_rate": source.sample_rate,
        "contexts": [c.detach().cpu() for c in source.contexts],
        "atoms": (
            [a.detach().cpu() for a in source.atoms]
            if source.atoms is not None
            else None
        ),
    }

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)
    logger.info(
        "saved source cache: %s (contexts=%d, atoms=%s)",
        path.name,
        len(payload["contexts"]),
        "none" if payload["atoms"] is None else len(payload["atoms"]),
    )
    return path


def load_encoded_source(source_id: str) -> EncodedSource:
    """Load an EncodedSource from the canonical cache.

    Raises FileNotFoundError if missing, or ValueError on a version mismatch.
    """
    path = source_cache_path(source_id)
    if not path.exists():
        raise FileNotFoundError(f"no cached source for {source_id!r} at {path}")

    payload = torch.load(path, map_location="cpu", weights_only=False)
    version = payload.get("version")
    if version != SOURCE_CACHE_VERSION:
        raise ValueError(
            f"source cache version mismatch for {source_id!r}: "
            f"got {version!r}, expected {SOURCE_CACHE_VERSION}"
        )

    return EncodedSource(
        source_id=payload["source_id"],
        contexts=list(payload["contexts"]),
        atoms=(list(payload["atoms"]) if payload.get("atoms") is not None else None),
        sample_rate=int(payload.get("sample_rate", 48000)),
    )


def encode_and_cache(
    engine: FlowInference,
    audio_path: Path,
    source_id: str,
    *,
    save_atoms: bool = False,
) -> EncodedSource:
    """Encode an audio file with SCAPES and persist it to the canonical cache."""
    if not audio_path.exists():
        raise FileNotFoundError(f"audio file does not exist: {audio_path}")

    logger.info("encoding source %r from %s", source_id, audio_path)
    audio_tensor = engine.load_audio_to_tensor(str(audio_path))
    atoms = engine.encode_audio_to_atoms(audio_tensor)
    contexts = engine.compute_context_track(atoms)

    source = EncodedSource(
        source_id=source_id,
        contexts=[c.detach().cpu() for c in contexts],
        atoms=([a.detach().cpu() for a in atoms] if save_atoms else None),
        sample_rate=engine.sr,
    )
    save_encoded_source(source)
    return source


def get_or_encode(
    engine: FlowInference,
    name: str,
    kind: str,
    *,
    audio_path: Optional[Path] = None,
    save_atoms: bool = False,
) -> EncodedSource:
    """Return a cached EncodedSource for the ``(name, kind)`` sound variant.

    The cache id is ``f"{kind}__{name}"`` so the short and long variants of a
    shared stem get distinct caches. `audio_path` defaults to
    ``assets/{kind}/{name}.wav``; pass it explicitly to point at uploads.
    """
    source_id = source_cache_key(name, kind)

    try:
        return load_encoded_source(source_id)
    except FileNotFoundError:
        logger.info("source cache miss for %r; encoding now", source_id)

    if audio_path is None:
        audio_path = audio_asset_path(name, kind)

    return encode_and_cache(engine, audio_path, source_id, save_atoms=save_atoms)
