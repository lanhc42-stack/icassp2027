from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from source_capture.scoring import attribute_tokens  # noqa: E402


class ScoringTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {"lowercase": True, "strip_punctuation": True}

    def test_common_words_are_ambiguous(self) -> None:
        result = attribute_tokens(
            "hello common lyric unknown",
            "hello common speech",
            "lyric common song",
            self.config,
        )
        self.assertEqual(result["n_speech"], 1)
        self.assertEqual(result["n_lyric"], 1)
        self.assertEqual(result["n_ambiguous"], 1)
        self.assertEqual(result["n_other"], 1)
        self.assertEqual(result["lir"], 0.5)
        self.assertEqual(result["scs"], 0.0)

    def test_reference_multiplicity_caps_matches(self) -> None:
        result = attribute_tokens("lyric lyric lyric", "speech", "lyric", self.config)
        self.assertEqual(result["n_lyric"], 1)
        self.assertEqual(result["n_other"], 2)

    def test_no_grounded_output(self) -> None:
        result = attribute_tokens("unrelated", "speech", "lyric", self.config)
        self.assertIsNone(result["lir"])
        self.assertIsNone(result["scs"])
        self.assertTrue(result["no_grounded_output"])


if __name__ == "__main__":
    unittest.main()

