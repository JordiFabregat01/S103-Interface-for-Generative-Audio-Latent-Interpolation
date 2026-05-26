"""Unit tests for overlap interpolation geometry (no SCAPES model required)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import unittest

import torch

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from inference.interpolation import (  # noqa: E402
    MIN_TIMELINE_SIZE_DYNAMIC,
    _cap_timeline_for_dynamic,
    _decoded_duration_sec,
    _dynamic_window,
    _timeline_size_for_duration,
    validate_overlap_anchors,
)

def _import_stitch_helpers():
    from inference.methods import (  # noqa: E402
        _crossfade_append,
        _stitch_segments,
        _trim_bridge_to_overlap,
    )

    return _crossfade_append, _stitch_segments, _trim_bridge_to_overlap


def _mock_engine(
    *,
    sr: int = 48000,
    hop_samples: int = 4800,
    segment_samples: int = 15360,
) -> SimpleNamespace:
    """Matches ATOMS_HOP_FRAMES=15, ATOMS_FRAMES=48 at 48 kHz."""
    return SimpleNamespace(sr=sr, hop_samples=hop_samples, segment_samples=segment_samples)


class TestTimelineSizeFormula(unittest.TestCase):
    def test_timeline_size_matches_decode_geometry(self):
        engine = _mock_engine()
        hop_sec = engine.hop_samples / engine.sr
        segment_sec = engine.segment_samples / engine.sr

        for target_sec in (0.35, 1.0, 2.0):
            new_n = _timeline_size_for_duration(engine, target_sec)
            new_decoded = _decoded_duration_sec(engine, new_n)
            self.assertGreaterEqual(new_decoded, target_sec)

        # Legacy round(duration/hop) could pick too few steps for sub-second targets.
        target_sec = 0.35
        legacy_n = max(1, int(round(target_sec / hop_sec)))
        legacy_decoded = (legacy_n - 1) * hop_sec + segment_sec
        new_n = _timeline_size_for_duration(engine, target_sec)
        self.assertGreaterEqual(_decoded_duration_sec(engine, new_n), target_sec)
        if legacy_decoded < target_sec:
            self.assertGreater(new_n, legacy_n)

    def test_single_step_minimum_when_shorter_than_one_segment(self):
        engine = _mock_engine()
        self.assertEqual(_timeline_size_for_duration(engine, 0.2), 1)
        self.assertAlmostEqual(_decoded_duration_sec(engine, 1), 0.32, delta=0.01)

    def test_dynamic_minimum_is_at_least_five_when_contexts_allow(self):
        self.assertGreaterEqual(MIN_TIMELINE_SIZE_DYNAMIC, 5)


class TestDynamicWindow(unittest.TestCase):
    def test_dynamic_window_raises_instead_of_padding(self):
        contexts = [torch.zeros(3), torch.zeros(3), torch.zeros(3)]
        with self.assertRaises(ValueError):
            _dynamic_window(contexts, anchor_atom=2, timeline_size=3, source_id="a")

    def test_cap_timeline_limits_to_available_contexts(self):
        contexts_a = [torch.zeros(8) for _ in range(8)]
        contexts_b = [torch.zeros(8) for _ in range(8)]
        capped = _cap_timeline_for_dynamic(20, contexts_a, contexts_b, 5, 0)
        self.assertEqual(capped, 3)


class TestOverlapAnchorValidation(unittest.TestCase):
    def test_rejects_overlap_with_both_anchors_zero(self):
        with self.assertRaises(ValueError):
            validate_overlap_anchors(-1.0, 0.0, 0.0)

    def test_allows_overlap_with_explicit_anchor(self):
        validate_overlap_anchors(-1.0, 8.0, 0.0)


class TestStitchHelpers(unittest.TestCase):
    def test_crossfade_blends_overlap_region(self):
        _crossfade_append, _, _ = _import_stitch_helpers()
        sr = 48000
        a = torch.ones(2, sr)
        b = torch.zeros(2, sr)
        fade = int(0.05 * sr)
        out = _crossfade_append(a, b, fade)
        self.assertEqual(out.shape[-1], 2 * sr - fade)
        mid = out[..., sr - fade // 2 : sr + fade // 2].mean()
        self.assertGreater(float(mid), 0.0)
        self.assertLess(float(mid), 1.0)

    def test_trim_bridge_center_crops_long_audio(self):
        _, _, _trim_bridge_to_overlap = _import_stitch_helpers()
        sr = 48000
        bridge = torch.randn(2, int(1.0 * sr))
        trimmed = _trim_bridge_to_overlap(bridge, 0.5, sample_rate=sr)
        self.assertEqual(trimmed.shape[-1], int(0.5 * sr))

    def test_stitch_segments_applies_crossfade(self):
        _, _stitch_segments, _ = _import_stitch_helpers()
        sr = 48000
        a = torch.ones(1, sr // 10)
        b = torch.zeros(1, sr // 10)
        out = _stitch_segments([a, b], sample_rate=sr, crossfade_sec=0.01)
        self.assertLess(out.shape[-1], 2 * (sr // 10))


if __name__ == "__main__":
    unittest.main()
