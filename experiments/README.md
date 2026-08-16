# Experiment artifacts

This directory contains both the formal E1-E8 experiment package and older
pilot artifacts. They must not be combined when reporting sample counts or
paper results.

## Formal experiment package

- `e1_e3/generated/e1|e2|e3/`: frozen manifests, model runs, scores, and
  summaries for E1-E3.
- `e4` through `e8`: development artifacts for the later experiments.
- `final_holdout/summaries/`: copies of the completed final-holdout summaries
  used by the result figures.
- `final_holdout/figures/`: figures generated directly from those summaries.

The executed formal ASR models for E1-E3 were Whisper large-v3 and Qwen3-ASR
1.7B. CTC support exists in the code, but no CTC result is present in the final
E1-E3 summaries and no paper claim may imply otherwise.

## Historical pilot artifacts

JSON files directly under `experiments/` and legacy scripts directly under
`src/` belong to an earlier exploratory phase. In particular,
`manifest_ts.json` describes an 800-stimulus, 50-track pilot manifest. The
corresponding raw CTC/Whisper/Qwen model-run files referenced by old notes are
not present here.

These historical files may be inspected for provenance or code reuse, but their
old narrative conclusions are not an evidence source for the final paper.

## Reporting rule

Every reported number must point to a machine-readable summary or be recomputed
from a manifest/run/score file. If no such artifact is present, do not claim the
result.
