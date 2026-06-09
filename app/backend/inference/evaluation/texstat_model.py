"""TexStat embedding model wrapped as a kadtk ``ModelLoader``.

kadtk has no TexStat model, so we vendor the reference implementation from
``cordutie/ddsp_textures`` (``modules/ddsp_textures/experiments/texstat``) and
expose its McDermott-Simoncelli summary statistics as a kadtk embedding model.

Wrapping it as a ``ModelLoader`` means ``KernelAudioDistance`` computes KAD over
these features with the *exact same* kernel/MMD used for KAD-CLAP, so the three
reported metrics are methodologically consistent.

Config mirrors ``experiments/texstat/fad.py::FAD_wrapper`` as used in the paper's
evaluation notebook (``frame_size = 44100`` at 44.1 kHz) -- see
``texdsp_and_noisebandnet_resynthesis_evaluation.ipynb`` in the submodule:
``FAD_wrapper(frame_size=44100, sampling_rate=44100)``.

Imported only during the scoring phase (it needs ``kadtk``); never imported by
the generation phase.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

from kadtk.model_loader import ModelLoader

# repo root from app/backend/inference/evaluation/texstat_model.py
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
# `import texstat.*` resolves against the experiments/ dir (texstat is the package)
_TEXSTAT_PARENT = _REPO_ROOT / "modules" / "ddsp_textures" / "experiments"


def _ensure_texstat_on_path() -> None:
    functions_py = _TEXSTAT_PARENT / "texstat" / "functions.py"
    filterbanks_py = _TEXSTAT_PARENT / "texstat" / "torch_filterbanks" / "filterbanks.py"
    if not functions_py.exists() or not filterbanks_py.exists():
        raise SystemExit(
            "TexStat sources not found under "
            f"{_TEXSTAT_PARENT / 'texstat'}.\n"
            "Initialise the submodule and its filterbanks:\n"
            "    git submodule update --init --recursive\n"
            "    git clone https://github.com/cordutie/torch_filterbanks.git "
            "modules/ddsp_textures/experiments/texstat/torch_filterbanks"
        )
    p = str(_TEXSTAT_PARENT)
    if p not in sys.path:
        sys.path.insert(0, p)


class TexStatModel(ModelLoader):
    """kadtk embedding model emitting TexStat (McDermott-Simoncelli) statistics.

    ``_get_embedding`` segments a waveform into ``frame_size`` frames and returns
    one statistics vector per frame, shape ``(n_frames, num_features)``.
    """

    def __init__(
        self,
        frame_size: int = 44100,
        sample_rate: int = 44100,
        n_filter_bank: int = 16,
        m_filter_bank: int = 6,
        n_moments: int = 4,
        downsampling_factor: int = 4,
        spectrum_lower_bound: float = 20.0,
        spectrum_higher_bound: float = 16000.0,
    ):
        # sub_statistics feature vector = S1 (moments) + S2 (env corr) + S3 (mod energy)
        num_features = (
            n_filter_bank * n_moments
            + (n_filter_bank * (n_filter_bank - 1)) // 2
            + n_filter_bank * m_filter_bank
        )
        super().__init__("texstat", num_features, sample_rate)

        self.frame_size = frame_size
        self.n_filter_bank = n_filter_bank
        self.m_filter_bank = m_filter_bank
        self.n_moments = n_moments
        self.downsampling_factor = downsampling_factor
        self.spectrum_lower_bound = spectrum_lower_bound
        self.spectrum_higher_bound = spectrum_higher_bound
        # FAD_wrapper's alpha weighting for S_1, truncated to n_moments
        self._alpha = torch.tensor([1.0, 1.0 / 10, 1.0 / 100, 1.0 / 1000])[:n_moments]

        self._feature_fn = None
        self._coch_fb = None
        self._mod_fb = None
        self._downsampler = None

    def load_model(self):
        _ensure_texstat_on_path()
        import torchaudio
        import texstat.torch_filterbanks.filterbanks as fb
        from texstat.functions import sub_statistics_mcds_feature_vector

        new_sr = self.sr // self.downsampling_factor
        new_frame_size = self.frame_size // self.downsampling_factor

        # Built on CPU (matching FAD_wrapper); the filterbanks hold their own
        # internal tensors, so we run the statistics on CPU regardless of the
        # KAD device and let kadtk do the kernel math on its device.
        self._coch_fb = fb.EqualRectangularBandwidth(
            self.frame_size,
            self.sr,
            self.n_filter_bank,
            self.spectrum_lower_bound,
            self.spectrum_higher_bound,
        )
        self._mod_fb = fb.Logarithmic(new_frame_size, new_sr, self.m_filter_bank, 10, new_sr // 4)
        self._downsampler = torchaudio.transforms.Resample(self.sr, new_sr)
        self._feature_fn = sub_statistics_mcds_feature_vector

    def load_wav(self, wav_file) -> np.ndarray:
        """Load mono audio resampled to ``self.sr`` (44.1 kHz for TexStat)."""
        import librosa

        wav, _ = librosa.load(str(wav_file), sr=self.sr, mono=True)
        return np.asarray(wav, dtype=np.float32)

    def _segment(self, audio: np.ndarray):
        fs = self.frame_size
        n_full = len(audio) // fs
        if n_full == 0:
            pad = np.zeros(fs, dtype=np.float32)
            pad[: len(audio)] = audio
            return [pad]
        return [audio[i * fs : (i + 1) * fs] for i in range(n_full)]

    def _get_embedding(self, audio: np.ndarray) -> torch.Tensor:
        if self._feature_fn is None:
            self.load_model()

        feats = []
        for seg in self._segment(np.asarray(audio, dtype=np.float32)):
            sig = torch.from_numpy(np.ascontiguousarray(seg)).float()
            with torch.no_grad():
                vec = self._feature_fn(
                    sig,
                    self._coch_fb,
                    self._mod_fb,
                    self._downsampler,
                    self.n_moments,
                    self._alpha,
                ).detach().cpu()
            # Mirror fad.py: drop numerically unstable segments.
            if torch.isnan(vec).any() or torch.isinf(vec).any():
                continue
            feats.append(vec)

        if not feats:  # keep a row so downstream stacking never sees an empty set
            feats.append(torch.zeros(self.num_features))
        return torch.stack(feats, dim=0)  # (n_frames, num_features)
