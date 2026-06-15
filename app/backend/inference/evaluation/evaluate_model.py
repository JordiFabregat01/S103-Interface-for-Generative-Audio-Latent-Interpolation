"""Evaluate the trained SCAPES model with audio-distribution distances.

For every source sound we resynthesize ~60s of the *same* texture and then
measure how close the generated audio's embedding distribution is to the
original's. Three distances are reported, each from the Kernel Audio Distance
Toolkit (``kadtk``):

    - FAD-VGGish   : Frechet distance over VGGish embeddings.
    - KAD-CLAP     : Kernel (MMD) distance over CLAP-2023 embeddings.
    - KAD-TexStat  : Kernel (MMD) distance over TexStat embeddings.

Lower is better for all three. We run on the *full* dataset (no cherry-picking);
scores would improve if only the best generations were kept.

The script is split into two phases that talk to each other only through WAV
files on disk, because they need different Python environments:

    generate  -> project venv  (torch / msclap / SCAPES)  : writes WAVs
    score     -> kadtk venv    (kadtk / openpyxl)          : reads WAVs, writes xlsx

Usage (run with CWD = app/backend):

    # phase 1, in the project venv (activate .venv first, or call its python)
    python -m inference.evaluation.evaluate_model --phase generate

    # phase 2, in a venv with `pip install kadtk openpyxl`
    python -m inference.evaluation.evaluate_model --phase score

    # both at once (only if one env happens to satisfy both)
    python -m inference.evaluation.evaluate_model --phase all

Smoke testing: add ``--limit 1`` or ``--only <stem>`` to work on a single sound.

----------------------------------------------------------------------------
kadtk API note: the scoring phase uses only ``kadtk``'s stable, documented
entry points -- ``get_all_models()`` (model registry) and the
``score(baseline_dir, eval_dir)`` method of ``FrechetAudioDistance`` /
``KernelAudioDistance`` -- plus ``cache_embedding_files`` for caching. Per-class
scores are computed by pointing ``score`` at one-file temp dirs. If the
installed kadtk version renames these symbols, only ``_load_kadtk`` /
``_make_distance`` / ``_score_dirs`` below need adjusting. If a model name
(``vggish`` / ``clap-2023`` / ``texstat``) is not in the registry, the error
lists the available names so you can fix ``METRICS`` without hunting.
----------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

# --- import bootstrap: make `import inference.*` work regardless of CWD -------
BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[2]  # app/backend
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("evaluate_model")

# Default output layout (under this package dir, gitignored).
DEFAULT_OUT_DIR = pathlib.Path(__file__).resolve().parent / "out"
REFERENCE_SUBDIR = "reference"
GENERATED_SUBDIR = "generated"
DEFAULT_EXCEL_NAME = "eval_results.xlsx"

DEFAULT_DURATION_SEC = 60.0
DEFAULT_NFE = 32
DEFAULT_SEED = 0
COMMON_SR = 48000  # reference & generated are written identically to avoid confounds


@dataclass(frozen=True)
class Metric:
    """One column in the report: an embedding model + a distance kind."""

    column: str      # report column header
    model: str       # kadtk model registry name
    kind: str        # "fad" (Frechet) or "kad" (Kernel / MMD)


# These three are exactly what the eval plan calls for. Model names follow the
# kadtk registry; adjust here if the installed version uses different keys.
METRICS: List[Metric] = [
    Metric(column="FAD-VGGish", model="vggish", kind="fad"),
    Metric(column="KAD-CLAP", model="clap-2023", kind="kad"),
    Metric(column="KAD-TexStat", model="texstat", kind="kad"),
]

OVERALL_ROW_LABEL = "OVERALL (pooled)"


# ===========================================================================
# Shared helpers
# ===========================================================================
def _discover_sounds(
    data_dir: pathlib.Path,
    *,
    only: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[pathlib.Path]:
    """Return the WAV files to evaluate, discovered at runtime.

    The dataset is intentionally *not* hardcoded (no AudioElement enum): the set
    of sounds is whatever ``*.wav`` files live in ``data_dir``, so adding,
    removing, or renaming sources just works. ``only`` selects a single stem and
    ``limit`` truncates the (sorted) list -- both for smoke tests.
    """
    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(f"data-dir does not exist or is not a directory: {data_dir}")

    wavs = sorted(p for p in data_dir.iterdir() if p.suffix.lower() == ".wav")
    if only is not None:
        wavs = [p for p in wavs if p.stem == only]
        if not wavs:
            raise FileNotFoundError(f"--only {only!r} matched no WAV in {data_dir}")
    if limit is not None:
        wavs = wavs[: max(0, limit)]

    if not wavs:
        raise FileNotFoundError(f"no .wav files found in data-dir: {data_dir}")

    logger.info("found %d sound(s) in %s", len(wavs), data_dir)
    return wavs


def _paired_stems(reference_dir: pathlib.Path, generated_dir: pathlib.Path) -> List[str]:
    """Stems present in BOTH dirs (robust to dataset changes / phase-1 gaps)."""
    ref = {p.stem for p in reference_dir.glob("*.wav")}
    gen = {p.stem for p in generated_dir.glob("*.wav")}
    paired = sorted(ref & gen)
    for stem in sorted(ref - gen):
        logger.warning("reference %r has no generated counterpart; skipping", stem)
    for stem in sorted(gen - ref):
        logger.warning("generated %r has no reference counterpart; skipping", stem)
    return paired


# ===========================================================================
# Phase 1: generation (project venv)
# ===========================================================================
def _write_mono_48k(samples, sr: int, path: pathlib.Path) -> None:
    """Write a 1-D float array as 48k mono PCM-16, clamped to [-1, 1]."""
    import numpy as np
    import soundfile as sf

    arr = np.asarray(samples, dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=0)  # downmix channels -> mono
    np.clip(arr, -1.0, 1.0, out=arr)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), arr, sr, subtype="PCM_16")


def _build_reference(orig_path: pathlib.Path, dest: pathlib.Path, duration_sec: float) -> int:
    """Trim the original to <= duration and write it in the common format.

    Returns the number of samples written.
    """
    import librosa

    waveform, _ = librosa.load(str(orig_path), sr=COMMON_SR, mono=True)
    n = int(round(duration_sec * COMMON_SR))
    waveform = waveform[:n]
    _write_mono_48k(waveform, COMMON_SR, dest)
    return int(waveform.shape[-1])


def phase_generate(args: argparse.Namespace) -> None:
    import torch

    from inference.methods import get_inference_engine
    from inference.scapes_runtime import run_resynthesis_pipeline

    out_dir = args.out_dir
    reference_dir = out_dir / REFERENCE_SUBDIR
    generated_dir = out_dir / GENERATED_SUBDIR
    reference_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)

    wavs = _discover_sounds(args.data_dir, only=args.only, limit=args.limit)

    torch.manual_seed(args.seed)  # generation samples white noise; pin it
    engine = get_inference_engine()
    logger.info("engine ready (sr=%d); resynthesizing %d sound(s)", engine.sr, len(wavs))

    n_ok = 0
    for i, orig_path in enumerate(wavs, start=1):
        stem = orig_path.stem
        ref_path = reference_dir / f"{stem}.wav"
        gen_path = generated_dir / f"{stem}.wav"
        t0 = time.time()
        try:
            _build_reference(orig_path, ref_path, args.duration)
            with torch.no_grad():
                wav = run_resynthesis_pipeline(
                    engine,
                    str(orig_path),
                    duration=args.duration,
                    play=False,
                    save_path=None,
                    TF=False,
                    NFE=args.nfe,
                    context_static=False,
                    decode_method="ola_smooth",
                )
            _write_mono_48k(wav.detach().cpu().float().numpy(), engine.sr, gen_path)
            n_ok += 1
            logger.info("[%d/%d] %s done in %.1fs", i, len(wavs), stem, time.time() - t0)
        except Exception:  # one bad sound must not abort the batch
            logger.exception("[%d/%d] %s FAILED; continuing", i, len(wavs), stem)
            # leave any partial files out of the way
            for p in (ref_path, gen_path):
                if p.exists():
                    try:
                        p.unlink()
                    except OSError:
                        pass

    logger.info(
        "generation complete: %d/%d ok -> %s and %s",
        n_ok,
        len(wavs),
        reference_dir,
        generated_dir,
    )


# ===========================================================================
# Phase 2: scoring (kadtk venv)
# ===========================================================================
def _load_kadtk():
    """Import the kadtk pieces we use, with a clear message if missing."""
    try:
        from kadtk.fad import FrechetAudioDistance
        from kadtk.kad import KernelAudioDistance
        from kadtk.fad_batch import cache_embedding_files
        from kadtk.model_loader import get_all_models
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "Scoring needs `kadtk` (Kernel Audio Distance Toolkit). Install it in a "
            "dedicated venv to avoid clashing with the project's pinned deps:\n"
            "    pip install kadtk openpyxl\n"
            f"(import error: {exc})"
        )
    return FrechetAudioDistance, KernelAudioDistance, cache_embedding_files, get_all_models


def _resolve_models(get_all_models) -> Dict[str, object]:
    """Map model name -> embedding model instance.

    kadtk-native models come from its registry; ``texstat`` is our custom
    ``ModelLoader`` (not in kadtk), injected here so the METRICS loop treats all
    three uniformly.
    """
    models = {m.name: m for m in get_all_models()}

    # Custom, non-kadtk embedding models.
    from inference.evaluation.texstat_model import TexStatModel

    models["texstat"] = TexStatModel()

    missing = [m.model for m in METRICS if m.model not in models]
    if missing:
        raise SystemExit(
            f"embedding model(s) not found: {missing}\n"
            f"Available models: {sorted(models)}\n"
            "Edit METRICS in this file to match the installed kadtk's names."
        )
    return models


def _make_distance(kind: str, model, device: str, FrechetAudioDistance, KernelAudioDistance):
    # kadtk's FrechetAudioDistance / KernelAudioDistance both require a device arg.
    if kind == "fad":
        return FrechetAudioDistance(model, device)
    if kind == "kad":
        return KernelAudioDistance(model, device)
    raise ValueError(f"unknown metric kind: {kind!r}")


def _score_dirs(distance, baseline_dir: pathlib.Path, eval_dir: pathlib.Path) -> float:
    """Distribution distance between all WAVs in two dirs (kadtk stable API)."""
    return float(distance.score(str(baseline_dir), str(eval_dir)))


def _count_windows(wav_path: pathlib.Path, cache_embedding_files, model) -> Optional[int]:
    """Best-effort count of embedding windows for one file (for the report).

    Caches the embedding (cheap on re-run) and reads its leading dimension.
    Returns None if the cached layout can't be located -- the column is
    informational only, so a miss is non-fatal.
    """
    import numpy as np

    try:
        from kadtk.utils import get_cache_embedding_path

        cache_path = pathlib.Path(get_cache_embedding_path(model.name, wav_path))
        if not cache_path.exists():
            cache_embedding_files(str(wav_path.parent), model)
        if cache_path.exists():
            return int(np.load(cache_path).shape[0])
    except Exception:  # informational only
        logger.debug("window count unavailable for %s", wav_path, exc_info=True)
    return None


def phase_score(args: argparse.Namespace) -> None:
    import pandas as pd

    (
        FrechetAudioDistance,
        KernelAudioDistance,
        cache_embedding_files,
        get_all_models,
    ) = _load_kadtk()

    out_dir = args.out_dir
    reference_dir = out_dir / REFERENCE_SUBDIR
    generated_dir = out_dir / GENERATED_SUBDIR
    for d in (reference_dir, generated_dir):
        if not d.exists() or not any(d.glob("*.wav")):
            raise SystemExit(
                f"missing WAVs in {d}. Run `--phase generate` first (project venv)."
            )

    stems = _paired_stems(reference_dir, generated_dir)
    if not stems:
        raise SystemExit("no reference/generated pairs to score.")
    logger.info("scoring %d paired sound(s)", len(stems))

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("scoring device: %s", device)

    models = _resolve_models(get_all_models)

    # rows keyed by sound stem; each row accumulates one value per metric column.
    rows: Dict[str, Dict[str, object]] = {stem: {"sound": stem} for stem in stems}
    overall: Dict[str, object] = {"sound": OVERALL_ROW_LABEL}

    # window counts (informational), filled lazily from the first metric's model.
    for metric in METRICS:
        model = models[metric.model]
        logger.info("metric %s: caching embeddings (%s)...", metric.column, metric.model)
        cache_embedding_files(str(reference_dir), model)
        cache_embedding_files(str(generated_dir), model)

        distance = _make_distance(
            metric.kind, model, device, FrechetAudioDistance, KernelAudioDistance
        )

        # Overall (pooled): all reference WAVs vs all generated WAVs.
        try:
            overall[metric.column] = _score_dirs(distance, reference_dir, generated_dir)
            logger.info("  overall %s = %s", metric.column, overall[metric.column])
        except Exception:
            logger.exception("  overall %s failed", metric.column)
            overall[metric.column] = float("nan")

        # Per-class: one-file temp dirs so we reuse the same stable score() API.
        for stem in stems:
            ref_file = reference_dir / f"{stem}.wav"
            gen_file = generated_dir / f"{stem}.wav"
            try:
                with tempfile.TemporaryDirectory() as tref, tempfile.TemporaryDirectory() as tgen:
                    tref_p = pathlib.Path(tref)
                    tgen_p = pathlib.Path(tgen)
                    shutil.copy2(ref_file, tref_p / ref_file.name)
                    shutil.copy2(gen_file, tgen_p / gen_file.name)
                    cache_embedding_files(str(tref_p), model)
                    cache_embedding_files(str(tgen_p), model)
                    rows[stem][metric.column] = _score_dirs(distance, tref_p, tgen_p)
            except Exception:
                logger.exception("  %s %s failed", stem, metric.column)
                rows[stem][metric.column] = float("nan")

    # window counts from the first metric's model (best-effort, informational)
    first_model = models[METRICS[0].model]
    for stem in stems:
        rows[stem]["n_ref_windows"] = _count_windows(
            reference_dir / f"{stem}.wav", cache_embedding_files, first_model
        )
        rows[stem]["n_gen_windows"] = _count_windows(
            generated_dir / f"{stem}.wav", cache_embedding_files, first_model
        )

    # ---- assemble DataFrame ----
    columns = ["sound", "n_ref_windows", "n_gen_windows"] + [m.column for m in METRICS]
    df = pd.DataFrame([rows[stem] for stem in stems])
    df = df.reindex(columns=columns)
    overall_df = pd.DataFrame([overall]).reindex(columns=columns)
    df = pd.concat([df, overall_df], ignore_index=True)

    excel_path = out_dir / args.excel_name
    note = pd.DataFrame(
        {
            "note": [
                "Lower is better for all three metrics.",
                "Scores reflect the FULL dataset (no cherry-picking); "
                "selecting best generations would lower the distances.",
                "FAD = Frechet Audio Distance; KAD = Kernel Audio Distance (MMD).",
                "Per-class FAD over a single ~60s file is noisy (few windows for the "
                "covariance); the pooled OVERALL row is the headline number.",
            ]
        }
    )
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="results")
        note.to_excel(writer, index=False, sheet_name="notes")

    logger.info("wrote %s", excel_path)
    # echo to console too
    try:
        logger.info("\n%s", df.to_string(index=False))
    except Exception:
        pass


# ===========================================================================
# CLI
# ===========================================================================
def _default_data_dir() -> pathlib.Path:
    from inference.constants import ASSETS_DIR

    return ASSETS_DIR


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate the SCAPES model with FAD-VGGish, KAD-CLAP, KAD-TexStat."
    )
    p.add_argument(
        "--phase",
        choices=["generate", "score", "all"],
        required=True,
        help="generate (project venv) | score (kadtk venv) | all",
    )
    p.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=None,
        help="dir of source WAVs to evaluate (default: inference assets dir).",
    )
    p.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=DEFAULT_OUT_DIR,
        help=f"output dir for reference/, generated/, and the xlsx (default: {DEFAULT_OUT_DIR}).",
    )
    p.add_argument("--excel-name", default=DEFAULT_EXCEL_NAME, help="xlsx filename in out-dir.")
    p.add_argument("--duration", type=float, default=DEFAULT_DURATION_SEC, help="seconds to generate/trim.")
    p.add_argument("--nfe", type=int, default=DEFAULT_NFE, help="ODE solver steps for generation.")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="torch seed for reproducible generation.")
    p.add_argument("--only", default=None, help="only evaluate this single sound stem (smoke test).")
    p.add_argument("--limit", type=int, default=None, help="evaluate at most N sounds (smoke test).")
    p.add_argument(
        "--device",
        default=None,
        help="scoring device for kadtk (cuda|cpu); default: auto-detect.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.out_dir = args.out_dir.resolve()
    if args.phase in ("generate", "all"):
        # data-dir default is resolved lazily so `score` doesn't need the project deps.
        args.data_dir = (args.data_dir or _default_data_dir())
        phase_generate(args)
    if args.phase in ("score", "all"):
        phase_score(args)


if __name__ == "__main__":
    main()
