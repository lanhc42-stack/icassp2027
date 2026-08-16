# Final-holdout analysis freeze — 2026-08-15

> **Protocol, not result.** This file records decisions frozen before holdout
> access. It does not state which hypotheses passed. Actual scope and outcomes
> are indexed in `EXPERIMENT_LEDGER.md` from machine-readable artifacts.

This file freezes the analysis policy before opening the track-level final holdout.
Development results have been inspected; no thresholds, feature sets, primary contrasts,
or missing-output rules may be changed after the holdout is opened. Any later analysis is
labelled exploratory.

## Scope

The nine paper experiments are E1, E2, E3, E4a, E4b, E5, E6, E7, and E8. E4c decoder
state patching, E5b same-person sung/spoken lyrics, and the CTC architecture extension are
optional follow-ups and do not block this experiment package.

## Frozen primary tests

| Experiment | Primary test | Support criterion |
|---|---|---|
| E1 | At −10 dB, onset vocal attenuation versus the mean of middle, offset, and distributed positions | track-bootstrap 95% CI excludes zero for both lower LIR and higher TSR |
| E2 | duration/dose trend at −10 and −5 dB | monotone reduction of LIR with increasing duration; attenuation saturation is reported rather than forced into a threshold claim |
| E3 | post-switch SCS relative to its matched static comparator | S→L speech-history effect is negative and grows with history; L→S asymmetry is reported |
| E4a | `SCS(lyric-prefix) − SCS(speech-prefix)` on boundary samples | positive track-bootstrap CI; available-case and SCS=0-for-no-grounded-output analyses must agree in sign |
| E4b | `carry-natural − reset` and `carry-counterfactual − carry-natural` within the identical B chunk | effects follow the source identity of the carried history; no-grounded-output rate is a separate endpoint |
| E5 | intact versus 500 ms and 100 ms shuffle | solo lyric recall falls with stronger disruption and mix LIR falls with it; reverse/instrumental are secondary controls |
| E6 | normalized onset patch effect versus middle/offset across layer | onset is largest in both directions; report raw SCS, median ratio, all ratios, and ratios restricted to `|full-patch SCS effect| ≥ 0.10` |
| E7 | held-out token NLL | compare M0/M1/M2 with the same binomial observation model, ridge penalty, and six-parameter budget; a loss by M2 is a valid negative result |
| E8 | actual onset/full/end enhancer versus baseline, with oracle onset as upper bound | actual onset improves LIR/TSR more than equal-budget end; report RTF and onset share of full gain |

## Frozen handling rules

- Track is the bootstrap and split unit; crops and speech utterances are repeated measures.
- SCS is undefined when no output token can be grounded. Both available-case SCS and a
  prespecified zero-filled sensitivity analysis are reported, together with the raw
  no-grounded-output rate.
- E6 is an explicitly stratified mechanism subset. Self patch must have zero or negligible
  SCS deviation; unstable full-patch denominators are not silently discarded.
- E5 instrumental matching is a secondary control. Its measured residual long-term spectral
  mismatch is reported; the lexical claim rests on the intact-to-shuffle gradient.
- E7 features and regularization are frozen. No post-holdout feature or threshold search is
  allowed.
- E8 uses the official DeepFilterNet release with delay compensation. Oracle stem ducking is
  never described as an actual deployable front end.

## Holdout opening rule

The holdout may be opened once E8 code passes a one-pair smoke test and all development
scripts and summaries have been archived. Holdout execution is one pass; failures caused by
software or corrupt files may be rerun without changing the experiment definition.
