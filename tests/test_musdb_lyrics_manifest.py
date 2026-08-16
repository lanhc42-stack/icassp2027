from pathlib import Path

from tools.build_musdb_lyrics_manifest import _read_onsets


def test_read_onsets_accepts_utf8_bom(tmp_path: Path) -> None:
    source = tmp_path / "utf8_align.csv"
    source.write_bytes("\ufeff0.5,café\n".encode("utf-8"))

    assert _read_onsets(source) == [(0.5, "café")]


def test_read_onsets_falls_back_to_cp1252_strictly(tmp_path: Path) -> None:
    source = tmp_path / "cp1252_align.csv"
    source.write_bytes("1.25,l’amour\n".encode("cp1252"))

    assert _read_onsets(source) == [(1.25, "l’amour")]
