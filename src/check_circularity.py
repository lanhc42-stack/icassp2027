"""
Is r(intelligibility, leak) real, or an artifact of the metric?

leak() counts hyp words that are in the lyric reference. The lyric reference IS
the vocals-alone transcript, so a track with few transcribable words has a tiny
reference and is mechanically bounded near zero leak.

Control: a reference-FREE leak measure -- words emitted that are absent from the
speech reference. It cannot mechanically depend on the lyric reference size.
If intelligibility still predicts that, the effect is real.
"""
import json
import re
import string
from collections import defaultdict
from pathlib import Path

RUNS = Path("/opt/tiger/icassp2027/work/runs")
MODELS = ["ctc_ts", "whisper_large_v3_ts"]


def words(s):
    if not s:
        return []
    s = s.lower().translate(str.maketrans("", "", string.punctuation))
    return [w for w in re.split(r"\s+", s) if w]


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = sum((a - mx) ** 2 for a in xs) ** 0.5
    dy = sum((b - my) ** 2 for b in ys) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


data = {m: json.loads((RUNS / f"{m}.json").read_text()) for m in MODELS}
lyric_ref, solo_len = defaultdict(set), defaultdict(int)
for m, rows in data.items():
    for r in rows:
        if r["cond"] == "vocals_alone" and r["track"]:
            w = words(r["hyp"])
            lyric_ref[r["track"]] |= set(w)
            solo_len[r["track"]] = max(solo_len[r["track"]], len(w))

for m, rows in data.items():
    ref_based, ref_free, wer_like = defaultdict(list), defaultdict(list), defaultdict(list)
    base = {}
    for r in rows:
        if r["cond"] == "speech_alone":
            base[r["utt"]] = r["hyp"]
    for r in rows:
        if r["cond"] != "C2_vocals" or r["snr"] != -10:
            continue
        h = words(r["hyp"])
        sp = set(words(r["ref"]))
        ly = lyric_ref.get(r["track"], set())
        if h:
            ref_based[r["track"]].append(len([w for w in h if w in ly and w not in sp]) / len(h))
            ref_free[r["track"]].append(len([w for w in h if w not in sp]) / len(h))
        # recall of the speech content: how much of the utterance survived
        wer_like[r["track"]].append(
            len([w for w in sp if w in set(h)]) / max(len(sp), 1))

    tracks = sorted(ref_based)
    x = [solo_len[t] for t in tracks]
    a = [sum(ref_based[t]) / len(ref_based[t]) for t in tracks]
    b = [sum(ref_free[t]) / len(ref_free[t]) for t in tracks]
    c = [sum(wer_like[t]) / len(wer_like[t]) for t in tracks]
    print(f"\n=== {m} (n={len(tracks)} tracks, SNR -10) ===")
    print(f"  r(intelligibility, ref-BASED leak) = {pearson(x, a):+.3f}   <- may be circular")
    print(f"  r(intelligibility, ref-FREE  leak) = {pearson(x, b):+.3f}   <- control")
    print(f"  r(intelligibility, speech recall)  = {pearson(x, c):+.3f}   <- speech destroyed?")
    print(f"  mean ref-free leak = {sum(b)/len(b):.3f}   mean speech recall = {sum(c)/len(c):.3f}")
