import json
from collections import defaultdict
from pathlib import Path

rows = json.loads(Path("/opt/tiger/icassp2027/work/pilot_results.json").read_text())

print("=== what the model transcribes on SINGING ALONE (no speech) ===")
for r in rows:
    if r["cond"] == "vocals_alone":
        print(f"  {r['track'][:34]:34s} -> {r['hyp'][:100]!r}")

print("\n=== speech-alone baseline (false-positive floor) ===")
agg = defaultdict(list)
for r in rows:
    if r["cond"] == "speech_alone":
        agg["speech_alone"].append(r["extra_ratio"])
for k, v in agg.items():
    print(f"  {k}: mean extra_ratio={sum(v)/len(v):.3f}  n={len(v)}")

print("\n=== mean extra-word ratio by condition x SNR ===")
cell = defaultdict(list)
for r in rows:
    if r["snr"] is None:
        continue
    cell[(r["cond"], r["snr"])].append(r["extra_ratio"])
conds = sorted({c for c, _ in cell})
snrs = sorted({s for _, s in cell})
print(f"{'cond':12s}" + "".join(f"{s:>+8d}" for s in snrs))
for c in conds:
    line = f"{c:12s}"
    for s in snrs:
        v = cell.get((c, s), [])
        line += f"{(sum(v)/len(v) if v else float('nan')):>8.2f}"
    print(line)

print("\n=== per-track C2_vocals (pure voice competition) ===")
cell2 = defaultdict(list)
for r in rows:
    if r["cond"] == "C2_vocals":
        cell2[(r["track"], r["snr"])].append(r["extra_ratio"])
tracks = sorted({t for t, _ in cell2})
print(f"{'track':36s}" + "".join(f"{s:>+8d}" for s in snrs))
for t in tracks:
    line = f"{t[:36]:36s}"
    for s in snrs:
        v = cell2.get((t, s), [])
        line += f"{(sum(v)/len(v) if v else float('nan')):>8.2f}"
    print(line)

print("\n=== most extreme leaks (extra_ratio >= 0.6) ===")
worst = sorted((r for r in rows if r.get("extra_ratio", 0) >= 0.6 and r["snr"] is not None),
               key=lambda r: -r["extra_ratio"])[:10]
for r in worst:
    print(f"  {r['cond']:11s} SNR{r['snr']:+4d} ratio={r['extra_ratio']:.2f} "
          f"{r['track'][:26]:26s} {r['hyp'][:70]!r}")
