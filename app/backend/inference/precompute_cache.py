"""Pre-compute encoded sources for every shipped asset.

Run as `python -m inference.precompute_cache` from the backend directory, or
import and call `precompute()` from a notebook.
"""

from __future__ import annotations

import logging

from inference.constants import source_cache_key
from inference.embeddings import _list_wav_files
from inference.methods import get_inference_engine
from inference.source_cache import encode_and_cache, source_cache_path

logger = logging.getLogger(__name__)


def precompute(*, save_atoms: bool = False, force: bool = False) -> None:
    """Encode a source cache for every wav in assets/short/ and assets/long/."""
    engine = get_inference_engine()
    for path, kind in _list_wav_files():
        source_id = source_cache_key(path.stem, kind)
        if not force and source_cache_path(source_id).exists():
            logger.info("skipping (already cached): %s", source_id)
            continue
        encode_and_cache(engine, path.resolve(), source_id, save_atoms=save_atoms)
        logger.info("cached source: %s", source_id)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    precompute()
