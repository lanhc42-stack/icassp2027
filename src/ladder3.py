"""Full three-rung ladder: CTC (no LM) -> Whisper (implicit) -> Qwen3-ASR (explicit, song-trained)."""
import json
import re
import string
import sys
from collections import defaultdict
from pathlib import Path

RUNS = Path("/opt/tiger/icassp2027/work/runs")
SUF = sys.argv[1] if len(sys.argv) > 1 else ""
LADDER = [("ctc", "no LM"), ("whisper_large_v3", "implicit LM"),
          ("qwen3_asr_1.7b", "explicit LLM + song")]
SNRS = [-10, -5, 0, 5, 10]


def words(s):
    if not s:
        return []
    s = s.lower().translate(str.maketrans("", "", string.punctuation))
    return [w for w in re.split(r"\s+", s) if w]


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = sum((a - mx) ** 2 for a in xs) ** 0.5
    dy = sum((b - my) ** 2 for b in ys) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


data = {}
for m, _ in LADDER:
    p = RUNS / f"{m}{SUF}.json"
    if p.exists():
        data[m] = json.loads(p.read_text())

lyric = defaultdict(set)
for rows in data.values():
    for r in rows:
        if r["cond"] == "vocals_alone" and r["track"]:
            lyric[r["track"]] |= set(words(r["hyp"]))


def leak(r):
    h = words(r["hyp"])
    if not h:
        return 0.0
    sp = set(words(r["ref"]))
    ly = lyric.get(r["track"], set())
    return len([w for w in h if w in ly and w not in sp]) / len(h)


def recall(r):
    sp, h = set(words(r["ref"])), set(words(r["hyp"]))
    return len([w for w in sp if w in h]) / max(len(sp), 1)


snrs = sorted({r["snr"] for rows in data.values() for r in rows if r["snr"] is not None})
print(f"=== LADDER: leak rate, C2_vocals {SUF or '(main set)'} ===")
print(f"{'model':26s}{'prior':22s}" + "".join(f"{s:>+9d}" for s in snrs))
means = {}
for m, label in LADDER:
    if m not in data:
        continue
    cell = defaultdict(list)
    for r in data[m]:
        if r["cond"] == "C2_vocals":
            cell[r["snr"]].append(leak(r))
    means[m] = {s: sum(v) / len(v) for s, v in cell.items() if v}
    print(f"{m:26s}{label:22s}" + "".join(f"{means[m].get(s, float('nan')):>9.3f}" for s in snrs))

print(f"\n=== step deltas (positive = upper rung leaks LESS) ===")
for (a, _), (b, _) in zip(LADDER, LADDER[1:]):
    if a in means and b in means:
        print(f"  {a} -> {b}: " + "".join(
            f"  {s:+d}:{means[a].get(s,0)-means[b].get(s,0):+.3f}" for s in snrs))

print(f"\n=== speech recall (how much target survives), C2_vocals ===")
print(f"{'model':26s}" + "".join(f"{s:>+9d}" for s in snrs))
for m, _ in LADDER:
    if m not in data:
        continue
    cell = defaultdict(list)
    for r in data[m]:
        if r["cond"] == "C2_vocals":
            cell[r["snr"]].append(recall(r))
    print(f"{m:26s}" + "".join(
        f"{(sum(cell[s])/len(cell[s]) if cell[s] else float('nan')):>9.3f}" for s in snrs))

print(f"\n=== floor (speech alone) ===")
for m, _ in LADDER:
    if m not in data:
        continue
    v = [leak(r) for r in data[m] if r["cond"] == "speech_alone"]
    rc = [recall(r) for r in data[m] if r["cond"] == "speech_alone"]
    if v:
        print(f"  {m:26s} leak={sum(v)/len(v):.3f}  recall={sum(rc)/len(rc):.3f}")

print(f"\n=== vocals-alone: transcript of pure singing ===")
for m, _ in LADDER:
    if m not in data:
        continue
    n = [len(words(r["hyp"])) for r in data[m] if r["cond"] == "vocals_alone"]
    if n:
        print(f"  {m:26s} mean words={sum(n)/len(n):5.1f}  silent(<2 words)={sum(1 for x in n if x<2)}/{len(n)}")

# per-track agreement across the ladder
pt = {}
for m, _ in LADDER:
    if m not in data:
        continue
    t = defaultdict(list)
    for r in data[m]:
        if r["cond"] == "C2_vocals" and r["snr"] == -10:
            t[r["track"]].append(leak(r))
    pt[m] = {k: sum(v) / len(v) for k, v in t.items()}
ms = [m for m, _ in LADDER if m in pt]
if len(ms) >= 2:
    print(f"\n=== per-track correlation @ -10 (n={len(set.intersection(*[set(pt[m]) for m in ms]))}) ===")
    common = sorted(set.intersection(*[set(pt[m]) for m in ms]))
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            r = pearson([pt[ms[i]][t] for t in common], [pt[ms[j]][t] for t in common])
            print(f"  {ms[i]:22s} vs {ms[j]:22s} r={r:+.3f}")

print(f"\n=== language flips (non-en) ===")
for m, _ in LADDER:
    if m not in data:
        continue
    fl, tot = defaultdict(int), defaultdict(int)
    for r in data[m]:
        if r["snr"] is None:
            continue
        tot[r["snr"]] += 1
        lg = (r.get("lang") or "").lower()
        if lg and lg not in ("en", "err", "english"):
            fl[r["snr"]] += 1
    print(f"  {m:26s}" + "".join(f"  {s:+d}:{fl[s]}/{tot[s]}" for s in snrs))
