# icassp2027

Research project targeting **ICASSP 2027** (Toronto, 16–21 May 2027).

## Hard deadline

| Milestone | Date |
|---|---|
| **Full paper submission** | **16 Sep 2026, 23:59:59 UTC-12** |
| Acceptance notification | 13 Jan 2027 |
| Final (camera-ready) paper | 27 Jan 2027 |
| Author registration | 10 Feb 2027 |

Format: 4 pages of technical content + 1 additional page for references only.
Verify against the official CfP before drafting — <https://2027.ieeeicassp.org/call-for-papers/>

## Topic

**Not yet locked.** See `docs/plan.md` for the decision and the back-planned schedule.

## Layout

```
src/          implementation
experiments/  one subdir per experiment: config + results + a short log
data/         datasets and audio (gitignored — keep out of git)
paper/        LaTeX source
docs/         plan, notes, related-work reading
```

## Conventions

- `data/` is gitignored. Never commit audio, checkpoints, or corpora.
- Every experiment gets a directory under `experiments/` with the exact config
  that produced its numbers. Results that can't be traced to a config don't go
  in the paper.
- Repo is **private** until the anonymity period ends.
