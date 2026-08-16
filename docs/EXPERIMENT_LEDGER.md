# Experiment ledger

This is a factual index, not a paper narrative. It is derived from the frozen
configuration, manifests, run files, and JSON summaries. When this file and a
machine artifact disagree, the machine artifact wins.

## Evidence authority

1. Final holdout summaries: `experiments/final_holdout/summaries/*.json`
2. Formal manifests and runs: `experiments/e1_e3/generated/e*/`
3. Frozen configuration: `experiments/e1_e3/generated/config.snapshot.yaml`
4. Pre-holdout rules: `docs/analysis-freeze-2026-08-15.md`

Planning documents, pilot Markdown, partial-run reports, and development-only
narratives are intentionally absent from the repository.

## Formal dataset scope

The formal configuration sets `n_tracks: 30`, `crops_per_track: 2`,
`n_speech_utterances: 5`, and `split_development_fraction: 0.70`.

| Experiment | Manifest rows | Unique tracks | Development | Holdout | Unique pairs |
|---|---:|---:|---:|---:|---:|
| E1 | 5,400 | 30 | 21 | 9 | 300 |
| E2 | 7,800 | 30 | 21 | 9 | 300 |
| E3 | 6,300 | 30 | 21 | 9 | 300 |

Whisper large-v3 and Qwen3-ASR 1.7B were executed. The final E1-E3 summaries
contain no CTC model.

The older `experiments/manifest_ts.json` contains an 800-row, 50-track pilot
manifest. The corresponding raw model-run files named in historical notes are
not present. Therefore the 50-track pilot must not be described as the formal
E1-E8 suite or used for final numerical claims from this repository.

## Final holdout audit against frozen criteria

The labels below apply the rules written before holdout access. They deliberately
avoid upgrading a partial or negative result into a positive claim.

| Experiment | Artifact-derived observation | Strict audit |
|---|---|---|
| E1 | At -10 dB, Qwen LIR and TSR CIs exclude zero in the favorable direction. Whisper TSR excludes zero, but Whisper LIR CI crosses zero. Effects are absent at 0 dB. | **Partial support**; the frozen cross-model, two-metric criterion is not fully met. |
| E2 | Qwen at -10 dB has dose correlations of -0.304 for LIR and +0.384 for TSR. Whisper LIR correlations are near zero, and the response is not monotonically cross-model. | **Frozen monotone-dose claim not supported.** |
| E3 | S-to-L history effects are negative for both models across tested durations; L-to-S effects are near zero or much smaller. Magnitude is not strictly monotone with duration, and symmetric 5 s order CIs cross zero. | **Directional asymmetry supported; monotone growth and generic order locking are not.** |
| E4a | Zero-filled boundary `SCS(lyric)-SCS(speech)` is +0.156 with CI [+0.023, +0.322]. Available-case evidence is weaker, and prompted conditions have 0.67-0.92 no-grounded-output rates. | **Inconclusive as a clean source-selection intervention; termination is a major confound.** |
| E4b | Some carry/reset and counterfactual contrasts exclude zero, but effects vary by direction and duration and often coincide with high no-grounded-output rates. | **Mixed support only.** |
| E5 | Intact mix LIR is 0.642; 500 ms shuffle is 0.354, 100 ms shuffle 0.225, and reverse 0.231. The paired shuffle/reverse LIR contrasts versus intact exclude zero. | **Intact-to-disruption result supported on the holdout.** |
| E6 | Development-only activation patching has zero self-patch deviation and a larger layer-averaged onset transfer than middle/offset, with strong direction asymmetry. No independent E6 holdout was run. | **Exploratory development evidence only.** |
| E7 | Held-out token NLL is 0.3155 for M0, 0.3154 for M1, and 0.3352 for M2. | **Negative result: M2 does not improve held-out fit.** |
| E8 | Oracle onset improves LIR and TSR. Actual onset-only enhancement has approximately zero LIR benefit and no TSR improvement; equal-budget end treatment is not clearly worse. | **Frozen practical onset-only criterion not supported.** |

## Final-holdout sizes for later experiments

| Experiment | Rows | Track scope |
|---|---:|---:|
| E4a | 144 | 7 tracks represented in the reported strata |
| E4b | 162 | 9 tracks |
| E5 | 270 | 9 tracks |
| E6 | 1,560 | 20 development tracks; no final holdout |
| E7 | 5,908 | 9 holdout tracks |
| E8 | 45 | 9 holdout tracks |

Repeated crops, speech utterances, conditions, and model outputs are not
independent tracks. Statistical uncertainty must be computed at track level.

## Claims not supported by the current artifacts

- The formal nine-experiment suite ran on all 50 MUSDB test tracks.
- A final CTC confirmation arm was completed.
- Two seconds is a universal or optimal intervention duration.
- Dose response is monotone across models and SNRs.
- Decoder history has been isolated cleanly from early termination.
- The six-parameter commitment model improves held-out prediction.
- Practical onset-only DeepFilterNet enhancement reproduces the oracle gain.
- E6 has independent holdout confirmation.

