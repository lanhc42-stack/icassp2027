# E1-E3 artifact archive

This directory contains the actual formal E1 position, E2 dose, and E3 history
experiment artifacts.

## Executed scope

The frozen configuration requested:

- 30 MUSDB tracks;
- 2 vocal crops per track;
- 5 LibriSpeech target utterances;
- a track-disjoint 70/30 split: 21 development tracks and 9 holdout tracks.

Actual manifest counts:

| Experiment | Stimuli | Tracks | Speech utterances | Speech-vocal pairs |
|---|---:|---:|---:|---:|
| E1 | 5,400 | 30 | 5 | 300 |
| E2 | 7,800 | 30 | 5 | 300 |
| E3 | 6,300 | 30 | 5 | 300 |

The executed ASR models were Whisper large-v3 and Qwen3-ASR 1.7B. Do not report
a final CTC arm: the summaries contain no CTC model.

## Artifact layout

```text
generated/
  config.snapshot.yaml
  e1|e2|e3/
    manifest.jsonl
    audio_checks.json
    runs/<model>.jsonl
    scores/<model>.jsonl
    summary.json
    summary.csv
```

`run` appends one JSONL row per completed sample and skips existing sample IDs,
so interrupted inference can resume. `summary.json` is the reporting source;
Markdown reports generated during partial or development-only runs were removed
because they were stale after final holdout completion.

## Reproduction entry points

- editable template: `configs/e1_e3.placeholder.yaml`;
- inference and analysis CLI: `python -m source_capture`;
- formal implementation: `src/source_capture/`.

Do not regenerate a manifest in place unless intentionally invalidating all
downstream runs, scores, and summaries that depend on its sample IDs.

