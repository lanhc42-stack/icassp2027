import numpy as np

from source_capture.e5 import _block_shuffle
from source_capture.e7 import MODEL_FEATURES, _segments_from_intervals, _window_lyric_advantage
from source_capture.e8 import _splice


def test_e5_block_shuffle_is_deterministic_and_length_preserving() -> None:
    audio = np.arange(16000, dtype=np.float32)
    first = _block_shuffle(audio, 1600, 10.0, 16000, 7, "crop")
    second = _block_shuffle(audio, 1600, 10.0, 16000, 7, "crop")
    assert len(first) == len(audio)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, audio)


def test_e7_onset_attenuation_changes_position_not_baseline_evidence() -> None:
    segments = _segments_from_intervals(-10.0, 20.0, [[0.0, 2.0]])
    onset = _window_lyric_advantage(segments, 0.0, 2.0)
    rest = _window_lyric_advantage(segments, 2.0, 10.0)
    assert np.isclose(onset, -10.0)
    assert np.isclose(rest, 10.0)


def test_e7_models_have_matched_feature_budget() -> None:
    assert {len(features) for features in MODEL_FEATURES.values()} == {5}


def test_e8_splice_changes_only_selected_window() -> None:
    base = np.zeros(1000, dtype=np.float32)
    enhanced = np.ones(200, dtype=np.float32)
    output = _splice(base, enhanced, 200, 400, 1000, 10.0)
    assert np.all(output[:200] == 0)
    assert np.all(output[400:] == 0)
    assert output[200] == 0
    assert output[210] == 1
    assert output[389] == 1
    assert output[399] == 0
