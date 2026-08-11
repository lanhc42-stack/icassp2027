"""
Track-variance analysis over 50 MUSDB test tracks.

Also tests a candidate explanation: does the INTELLIGIBILITY of the isolated
singing (proxied by how many words the model transcribes from vocals-alone)
predict how much that track leaks?
"""
import json
import re
import string
from collections import defaultdict
from pathlib import Path

RUNS = Path("/opt/tiger/icassp2027/work/runs")
MODELS = ["ctc_ts", "whisper_large_v3_ts"]
SNRS = [-10, -5, 0]


def words(s):
    if not s:
        return []
    s = s.lower().translate(str.maketrans("", "", string.punctuation))
    return [w for w in re.split(r"\s+", s) if w]


data = {m: json.loads((RUNS / f"{m}.json").read_text())
        for m in MODELS if (RUNS / f"{m}.json").exists()}
if len(data) < 2:
    print("missing:", [m for m in MODELS if m not in data])

lyric_ref, solo_len = defaultdict(set), defaultdict(int)
for m, rows in data.items():
    for r in rows:
        if r["cond"] == "vocals_alone" and r["track"]:
            w = words(r["hyp"])
            lyric_ref[r["track"]] |= set(w)
            solo_len[r["track"]] = max(solo_len[r["track"]], len(w))


def leak(hyp, ref, track):
    h = words(hyp)
    if not h:
        return 0.0
    sp, ly = set(words(ref)), lyric_ref.get(track, set())
    return len([w for w in h if w in ly and w not in sp]) / len(h)


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = sum((a - mx) ** 2 for a in xs) ** 0.5
    dy = sum((b - my) ** 2 for b in ys) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


per = {}
for m, rows in data.items():
    t = defaultdict(list)
    for r in rows:
        if r["cond"] == "C2_vocals" and r["snr"] == -10:
            t[r["track"]].append(leak(r["hyp"], r["ref"], r["track"]))
    per[m] = {k: sum(v) / len(v) for k, v in t.items()}

print("=== distribution of per-track leak @ SNR -10 (C2_vocals) ===")
for m in per:
    v = sorted(per[m].values())
    if not v:
        continue
    n = len(v)
    zero = sum(1 for x in v if x < 0.02)
    high = sum(1 for x in v if x > 0.40)
    print(f"\n-- {m} --  n={n}")
    print(f"   min={v[0]:.3f}  p25={v[n//4]:.3f}  median={v[n//2]:.3f} "
          f"p75={v[3*n//4]:.3f}  max={v[-1]:.3f}  mean={sum(v)/n:.3f}")
    print(f"   tracks with ~no leak (<0.02): {zero}/{n}   heavy leak (>0.40): {high}/{n}")

print("\n=== mean leak by SNR (all 50 tracks) ===")
for m, rows in data.items():
    cell = defaultdict(list)
    for r in rows:
        if r["cond"] == "C2_vocals":
            cell[r["snr"]].append(leak(r["hyp"], r["ref"], r["track"]))
    print(f"   {m:22s}" + "".join(
        f"  {s:+d}: {sum(cell[s])/len(cell[s]):.3f}" for s in SNRS if cell[s]))

if len(per) == 2:
    common = sorted(set(per[MODELS[0]]) & set(per[MODELS[1]]))
    a = [per[MODELS[0]][t] for t in common]
    b = [per[MODELS[1]][t] for t in common]
    print(f"\n=== do the two models leak on the SAME tracks? ===")
    print(f"   pearson r = {pearson(a, b):.3f}  over {len(common)} tracks")

print("\n=== does singing intelligibility predict leakage? ===")
for m in per:
    common = [t for t in per[m] if t in solo_len]
    xs = [solo_len[t] for t in common]
    ys = [per[m][t] for t in common]
    print(f"   {m:22s} r(vocals-alone word count, leak) = {pearson(xs, ys):+.3f}  n={len(common)}")

print("\n=== top / bottom tracks (mean of both models) ===")
if len(per) == 2:
    comb = {t: (per[MODELS[0]].get(t, 0) + per[MODELS[1]].get(t, 0)) / 2
            for t in set(per[MODELS[0]]) | set(per[MODELS[1]])}
    order = sorted(comb.items(), key=lambda kv: -kv[1])
    print("   -- highest --")
    for t, v in order[:6]:
        print(f"   {v:.3f}  solo_words={solo_len.get(t,0):3d}  {t[:44]}")
    print("   -- lowest --")
    for t, v in order[-6:]:
        print(f"   {v:.3f}  solo_words={solo_len.get(t,0):3d}  {t[:44]}")
