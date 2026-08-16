from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from source_capture.audio import (  # noqa: E402
    attenuation_envelope,
    common_scale,
    compose_unscaled,
    piecewise_snr_envelope,
    piecewise_gain_envelope,
    read_pcm16,
    rms,
    vocal_gain_for_snr,
    write_pcm16,
)


class AudioTest(unittest.TestCase):
    def test_snr_scaling(self) -> None:
        rng = np.random.default_rng(7)
        speech = rng.normal(size=16000).astype(np.float32) * 0.05
        vocal = rng.normal(size=16000).astype(np.float32) * 0.2
        for target in (-10.0, -5.0, 0.0, 10.0):
            gain = vocal_gain_for_snr(speech, vocal, target)
            realized = 20.0 * np.log10(rms(speech) / rms(vocal * gain))
            self.assertAlmostEqual(realized, target, places=5)

    def test_attenuation_envelope_has_smooth_contained_ramp(self) -> None:
        sample_rate = 1000
        envelope = attenuation_envelope(
            1000, sample_rate, [[0.2, 0.6]], attenuation_db=20, ramp_ms=50
        )
        self.assertEqual(float(envelope[0]), 1.0)
        self.assertAlmostEqual(float(envelope[200]), 1.0, places=6)
        self.assertAlmostEqual(float(envelope[250]), 0.1, places=5)
        self.assertAlmostEqual(float(envelope[550]), 0.1, places=5)
        self.assertAlmostEqual(float(envelope[599]), 1.0, places=5)
        self.assertEqual(float(envelope[600]), 1.0)

    def test_common_scale_is_shared_and_prevents_clipping(self) -> None:
        speech = np.full(100, 0.7, dtype=np.float32)
        vocal = np.full(100, 0.6, dtype=np.float32)
        first = compose_unscaled(speech, vocal, np.ones(100, dtype=np.float32))
        second = compose_unscaled(speech, vocal, np.full(100, 0.1, dtype=np.float32))
        scale = common_scale([first, second], 0.9)
        self.assertLess(scale, 1.0)
        self.assertLessEqual(float(np.max(np.abs(first * scale))), 0.900001)
        self.assertLessEqual(float(np.max(np.abs(second * scale))), 0.900001)

    def test_piecewise_envelope_realizes_local_snr(self) -> None:
        rng = np.random.default_rng(8)
        speech = rng.normal(size=16000).astype(np.float32) * 0.1
        vocal = rng.normal(size=16000).astype(np.float32) * 0.1
        envelope = piecewise_snr_envelope(
            speech, vocal, 16000, [(0.0, 0.5, 10.0), (0.5, 1.0, -10.0)], 0.0
        )
        first = 20.0 * np.log10(rms(speech[:8000]) / rms(vocal[:8000] * envelope[:8000]))
        second = 20.0 * np.log10(rms(speech[8000:]) / rms(vocal[8000:] * envelope[8000:]))
        # The envelope uses full-clip RMS by protocol; stationary synthetic noise
        # realizes each segment within a small finite-sample tolerance.
        self.assertAlmostEqual(first, 10.0, delta=0.2)
        self.assertAlmostEqual(second, -10.0, delta=0.2)

    def test_piecewise_gain_envelope_is_explicit(self) -> None:
        envelope = piecewise_gain_envelope(
            1000, 1000, [(0.0, 0.5, 0.2), (0.5, 1.0, 2.0)], 0.0
        )
        self.assertTrue(np.allclose(envelope[:500], 0.2))
        self.assertTrue(np.allclose(envelope[500:], 2.0))

    def test_pcm16_roundtrip(self) -> None:
        audio = np.linspace(-0.8, 0.8, 1000, dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio.wav"
            write_pcm16(path, audio, 16000)
            restored, sample_rate = read_pcm16(path)
        self.assertEqual(sample_rate, 16000)
        self.assertLess(float(np.max(np.abs(restored - audio))), 5e-5)


if __name__ == "__main__":
    unittest.main()
