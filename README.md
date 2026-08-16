# icassp2027

Research code and artifacts for a study of source capture in encoder-decoder ASR
under competing speech and singing.

## Source of truth

Do not infer experiment status or paper claims from planning notes, pilot names,
or prose summaries. Use this order:

1. `experiments/final_holdout/summaries/*.json` for final holdout aggregates;
2. `experiments/e1_e3/generated/e*/manifest.jsonl` and `runs/*.jsonl` for actual
   executed E1-E3 rows;
3. `experiments/e1_e3/generated/config.snapshot.yaml` for the frozen stimulus
   configuration;
4. `docs/analysis-freeze-2026-08-15.md` for the pre-holdout decision rules;
5. `docs/EXPERIMENT_LEDGER.md` for a factual index derived from those artifacts.

The formal controlled suite used **30 MUSDB tracks**, split by track into
**21 development and 9 final holdout tracks**. A separate historical pilot
manifest contains 50 tracks, but it is not the formal E1-E8 dataset and its raw
model-run files are not present in this repository.

## Repository layout

```text
configs/       editable templates; machine-specific *.local.yaml files are ignored
docs/          frozen protocol and artifact-derived experiment ledger
experiments/   manifests, raw outputs, scores, summaries, and figures
paper/         manuscript source
scripts/       resumable experiment launchers
src/           reusable implementation and historical pilot scripts
tests/         implementation tests
tools/         data utilities, server adapters, and plotting scripts
```

Audio, datasets, and model checkpoints must remain outside version control.
