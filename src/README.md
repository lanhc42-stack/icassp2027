# Source code index

## Formal experiment implementation

`source_capture/` contains the reusable package used by the formal E1-E8 suite:

- `prepare.py`, `generate.py`, `run.py`, `scoring.py`, and `analysis.py` implement
  E1-E3 preparation, inference, scoring, and analysis;
- `e4.py` through `e8.py` implement the later experiments;
- configuration templates live under `configs/`;
- resumable server entry points live under `scripts/` and `tools/`.

The final E1-E3 summaries contain only Whisper large-v3 and Qwen3-ASR 1.7B.
Although the package has a CTC backend, CTC was not executed as part of the
final E1-E3 result set.

## Historical pilot scripts

Standalone files such as `build_tracksweep.py`, `patch_window*.py`,
`patch_ctc.py`, and `table1_mitigation.py` predate the formal package. Their
names and inline comments are historical context, not final scientific claims.
Use them only when reproducing a specifically identified pilot artifact.

## Reproducibility rule

Machine-specific paths belong only in ignored `*.local.yaml` files. A result is
reportable only when its manifest, configuration, raw output, scoring output,
and machine-readable summary can be traced together.
