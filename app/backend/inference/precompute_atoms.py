"""Pre-compute and persist EnCodec atom tensors for every shipped asset.

Run from the backend directory:

    python -m inference.precompute_atoms            # skip already-cached atoms
    python -m inference.precompute_atoms --force    # re-encode everything
"""

from __future__ import annotations

import argparse
import logging
import time

from inference.constants import _get_audio_asset_path
from inference.methods import get_inference_engine
from inference.models import AudioElement
from inference.source_cache import (
    encode_and_cache,
    load_encoded_source,
    source_cache_path,
)

logger = logging.getLogger(__name__)


def _needs_atoms(source_id: str) -> bool:
    """Return True if this source has no cached atoms."""
    path = source_cache_path(source_id)
    if not path.exists():
        return True
    try:
        source = load_encoded_source(source_id)
        return source.atoms is None
    except Exception:
        return True


def precompute_atoms(*, force: bool = False) -> None:
    targets = [a for a in AudioElement if force or _needs_atoms(a.value)]

    if not targets:
        logger.info("all atom caches are up-to-date; nothing to do")
        return

    logger.info(
        "encoding atoms for %d/%d assets%s",
        len(targets),
        len(AudioElement),
        " (forced)" if force else "",
    )

    engine = get_inference_engine()

    for i, audio in enumerate(targets, 1):
        audio_path = _get_audio_asset_path(audio).resolve()
        t0 = time.perf_counter()
        encode_and_cache(engine, audio_path, audio.value, save_atoms=True)
        elapsed = time.perf_counter() - t0
        logger.info("[%d/%d] %s — %.1fs", i, len(targets), audio.value, elapsed)

    logger.info("done — encoded atoms for %d assets", len(targets))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Pre-compute EnCodec atom tensors for all audio assets."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-encode even if atom cache already exists.",
    )
    args = parser.parse_args()

    precompute_atoms(force=args.force)
