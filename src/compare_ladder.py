"""
Compare models on the prior-strength ladder over the identical clip set.

Leak metric: words in the hypothesis that appear in the LYRIC reference but NOT
in the speech reference. The lyric reference is built from the vocals-alone
transcripts (union across models, so it is model-independent). This is much
closer to a real LLR than "words not in the speech reference", which counts
ordinary ASR errors too.
"""
import json
import re
import string
from collections import defaultdict
from pathlib import Path

RUNS = Path("/opt/tiger/icassp2027/work/runs")
MODELS = ["ctc", "whisper_large_v3"]
SNRS = [-10, -5, 0, 5, 10]


def words(s):
    if not s:
        return []
    s = s.lower().translate(str.maketrans("", "", string.punctuation))
    return [w for w in re.split(r"\s+", s) if w]


data = {m: json.loads((RUNS / f"{m}.json").read_text())
        for m in MODELS if (RUNS / f"{m}.json").exists()}

# ---- model-independent lyric reference per track ----
lyric_ref = defaultdict(set)
for m, rows in data.items():
    for r in rows:
        if r["cond"] == "vocals_alone" and r["track"]:
            lyric_ref[r["track"]] |= set(words(r["hyp"]))

print("=== vocals-alone: what each model makes of pure singing ===")
for m, rows in data.items():
    print(f"\n-- {m} --")
    for r in rows:
        if r["cond"] == "vocals_alone":
            print(f"   {r['track'][:30]:30s} {r['hyp'][:78]!r}")


def leak(hyp, ref, track):
    h = words(hyp)
    if not h:
        return 0.0
    sp = set(words(ref))
    ly = lyric_ref.get(track, set())
    hits = [w for w in h if w in ly and w not in sp]
    return len(hits) / len(h)


print("\n\n=== LEAK RATE (lyric-attributable words / hyp words) ===")
for m, rows in data.items():
    floor = [leak(r["hyp"], r["ref"], None) for r in rows if r["cond"] == "speech_alone"]
    print(f"\n-- {m} --   speech-alone floor: {sum(floor)/max(len(floor),1):.3f}")
    cell = defaultdict(list)
    for r in rows:
        if r["snr"] is None:
            continue
        cell[(r["cond"], r["snr"])].append(leak(r["hyp"], r["ref"], r["track"]))
    conds = sorted({c for c, _ in cell})
    print(f"   {'cond':12s}" + "".join(f"{s:>+8d}" for s in SNRS))
    for c in conds:
        line = f"   {c:12s}"
        for s in SNRS:
            v = cell.get((c, s), [])
            line += f"{(sum(v)/len(v) if v else float('nan')):>8.3f}"
        print(line)

print("\n\n=== Delta0: CTC minus Whisper (C2_vocals) ===")
if len(data) == 2:
    per = {}
    for m in MODELS:
        cell = defaultdict(list)
        for r in data[m]:
            if r["cond"] == "C2_vocals":
                cell[r["snr"]].append(leak(r["hyp"], r["ref"], r["track"]))
        per[m] = {s: sum(v) / len(v) for s, v in cell.items()}
    print(f"   {'SNR':>6s} {'ctc':>8s} {'whisper':>8s} {'delta':>8s}")
    for s in SNRS:
        a, b = per["ctc"].get(s, 0), per["whisper_large_v3"].get(s, 0)
        print(f"   {s:>+6d} {a:>8.3f} {b:>8.3f} {a-b:>+8.3f}")

print("\n\n=== language flips (non-en outputs) ===")
for m, rows in data.items():
    flips = defaultdict(int)
    tot = defaultdict(int)
    for r in rows:
        if r["snr"] is None:
            continue
        tot[r["snr"]] += 1
        lg = (r.get("lang") or "").lower()
        if lg and lg not in ("en", "err"):
            flips[r["snr"]] += 1
    print(f"   {m:18s}" + "".join(f"  {s:+d}:{flips[s]}/{tot[s]}" for s in SNRS))

print("\n\n=== per-track spread at SNR -10 (C2_vocals) ===")
for m, rows in data.items():
    print(f"\n-- {m} --")
    pt = defaultdict(list)
    for r in rows:
        if r["cond"] == "C2_vocals" and r["snr"] == -10:
            pt[r["track"]].append(leak(r["hyp"], r["ref"], r["track"]))
    for t, v in sorted(pt.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        print(f"   {sum(v)/len(v):.3f}  {t[:46]}")
