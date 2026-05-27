"""Pre-compute encoded sources for every shipped asset.

Run as `python -m inference.precompute_cache` from the backend directory, or
import and call `precompute()` from a notebook.
"""

from __future__ import annotations

import logging

from inference.constants import _get_audio_asset_path
from inference.methods import get_inference_engine
from inference.models import AudioElement
from inference.source_cache import encode_and_cache, source_cache_path

logger = logging.getLogger(__name__)


def precompute(*, save_atoms: bool = False, force: bool = False) -> None:
    engine = get_inference_engine()
    for audio in AudioElement:
        if not force and source_cache_path(audio.value).exists():
            logger.info("skipping (already cached): %s", audio.value)
            continue
        audio_path = _get_audio_asset_path(audio).resolve()
        encode_and_cache(engine, audio_path, audio.value, save_atoms=save_atoms)
        logger.info("cached source: %s", audio.value)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    precompute()
