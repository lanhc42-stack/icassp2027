"""
Fig 3 step 2 -- linear probes over Whisper's own layers.

P1 "singing present"  : C2_vocals vs C1_rhythm, split BY TRACK (unseen songs).
P2 "which song"       : 50-way track id among C2 clips, split BY UTTERANCE.
                        Control = same probe on C1_rhythm (instrumental timbre
                        can also identify a track, so C2-vs-C1 is the contrast
                        that isolates vocal content).

Layer axis runs encoder 0..32 then decoder 0..32 (index 0 = embeddings).
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path("/opt/tiger/icassp2027")
d = np.load(ROOT / "work" / "probe_feats.npz", allow_pickle=True)
enc, dec = d["enc"], d["dec"]          # (N, L+1, D)
meta = json.loads(str(d["meta"]))
N = len(meta)
print(f"{N} clips, enc {enc.shape}, dec {dec.shape}")

tracks = sorted({m["track"] for m in meta})
utts = sorted({m["utt"] for m in meta})
tidx = {t: i for i, t in enumerate(tracks)}
cond = np.array([m["cond"] for m in meta])
trk = np.array([m["track"] for m in meta])
utt = np.array([m["utt"] for m in meta])
snr = np.array([m["snr"] for m in meta])


def fit(X, y, tr, te):
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)
    clf.fit(sc.transform(X[tr]), y[tr])
    return float(clf.score(sc.transform(X[te]), y[te]))


def layer_stack(i):
    """i in [0, 65] -> encoder 0..32 then decoder 0..32"""
    return enc[:, i] if i <= enc.shape[1] - 1 else dec[:, i - enc.shape[1]]


n_layers = enc.shape[1] + dec.shape[1]
print(f"probing {n_layers} layer positions "
      f"(encoder 0..{enc.shape[1]-1}, decoder 0..{dec.shape[1]-1})")

# ---- P1: singing present, held-out TRACKS ----
hold = set(tracks[::4])                      # 25% of songs unseen
tr1 = np.array([i for i in range(N) if trk[i] not in hold])
te1 = np.array([i for i in range(N) if trk[i] in hold])
y1 = (cond == "C2_vocals").astype(int)

# ---- P2: which song, held-out UTTERANCE ----
hold_u = utts[-1]
m_c2 = cond == "C2_vocals"
m_c1 = cond == "C1_rhythm"


def song_split(mask):
    tr = np.array([i for i in range(N) if mask[i] and utt[i] != hold_u])
    te = np.array([i for i in range(N) if mask[i] and utt[i] == hold_u])
    return tr, te


tr2, te2 = song_split(m_c2)
tr3, te3 = song_split(m_c1)
y2 = np.array([tidx[t] for t in trk])

print(f"P1 train/test {len(tr1)}/{len(te1)} (held-out songs: {len(hold)})")
print(f"P2 train/test {len(tr2)}/{len(te2)}   control {len(tr3)}/{len(te3)}"
      f"   chance={1/len(tracks):.3f}")

rows = []
for i in range(n_layers):
    X = layer_stack(i)
    a1 = fit(X, y1, tr1, te1)
    a2 = fit(X, y2, tr2, te2)
    a3 = fit(X, y2, tr3, te3)
    where = "enc" if i < enc.shape[1] else "dec"
    li = i if i < enc.shape[1] else i - enc.shape[1]
    rows.append({"pos": i, "block": where, "layer": li,
                 "singing_present": a1, "song_id_C2": a2, "song_id_C1": a3})
    print(f"  {where} L{li:02d}  singing={a1:.3f}  songC2={a2:.3f}  songC1={a3:.3f}",
          flush=True)

out = ROOT / "work" / "probe_results.json"
out.write_text(json.dumps(rows, indent=1))
print(f"\nwrote {out}")

print("\n=== summary ===")
for blk in ("enc", "dec"):
    r = [x for x in rows if x["block"] == blk]
    if not r:
        continue
    best = max(r, key=lambda x: x["song_id_C2"])
    print(f"{blk}: singing_present peak={max(x['singing_present'] for x in r):.3f}"
          f"  song_id_C2 peak={best['song_id_C2']:.3f} @L{best['layer']}"
          f"  last-layer song_id_C2={r[-1]['song_id_C2']:.3f}")
