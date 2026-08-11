# Plan — ICASSP 2027

Deadline: **16 Sep 2026, 23:59:59 UTC-12**. Started 11 Aug 2026 → **~36 days**.

That is a short runway. It rules out anything needing a new dataset, a new
annotation effort, or a model trained from scratch. What fits: a sharp claim on
top of infrastructure and data that already exist, with a clean baseline
comparison and one or two ablations.

## 1. Topic — TO DECIDE (blocking, needs to be settled this week)

Write the claim as one sentence before writing any code:

> "We show that ___ , which prior work does not, measured by ___ on ___ ."

Candidate directions (fill in / replace):

- [ ] …
- [ ] …

Selection criteria, in order:
1. Can the core result be produced with data already on hand?
2. Is there a published baseline to compare against, or must one be built?
3. Is the metric standard enough that reviewers won't argue about it?
4. Is it novel against the last two years of ICASSP/Interspeech?

## 2. Back-planned schedule

| By | Milestone | Done |
|---|---|---|
| **Aug 18** | Topic + one-sentence claim locked. Baseline reproduced, number recorded. Datasets in place. | [ ] |
| **Aug 28** | Main experiment produces the headline result. If it doesn't beat baseline by now, change the claim, not the deadline. | [ ] |
| **Sep 4** | All numbers frozen. Ablations done. Tables and figures generated from committed configs. | [ ] |
| **Sep 11** | Complete draft — every section written, nothing marked TODO. | [ ] |
| **Sep 14** | Internal read-through, revisions, page count within limit. | [ ] |
| **Sep 16** | Submit. Aim for a day early; the portal is unreliable at the deadline. | [ ] |

## 3. Risks

- **Negative result late.** The Aug 28 gate exists for this. Decide there,
  not in September.
- **Baseline is the real work.** If reproducing prior work eats two weeks,
  the contribution shrinks to fit what's left. Budget it explicitly.
- **Employer IP / approval.** If any data, model, or infrastructure is
  ByteDance's, start the internal publication-approval process now — it has
  its own lead time and is not something to discover on Sep 15.
- **Anonymity.** ICASSP review is double-blind. Keep this repo private, and
  don't preprint under a de-anonymizing name inside the review window without
  checking the policy.

## 4. Open questions

- Co-authors? Who, and what do they own?
- Compute: what's available, and is it reserved for the Aug 20 – Sep 4 window?
- Which track / EDICS category?
