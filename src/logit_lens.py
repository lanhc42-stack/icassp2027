"""
Fig 3 (primary) -- logit lens on Whisper's decoder.

No probe training, no memorisable labels: project every decoder layer's hidden
state through the model's own output head and read the token distribution.

If lyric tokens carry mass in early/mid decoder layers and lose it by the final
layer, suppression is localised inside the decoder -- within one model, so no
cross-model confound. That is the claim pilot 04 could only assert by
correlation.

For each clip we track, per decoder layer:
  p_lyric  mass on tokens belonging to that track's lyrics (and not the speech)
  p_speech mass on tokens of the speech reference (and not the lyrics)
"""
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

ROOT = Path("/opt/tiger/icassp2027")
DATA = ROOT / "data"
STEMS = DATA / "musdb16k"
MODEL = ROOT / "models" / "whisper-large-v3-hf"
RUNS = ROOT / "work" / "runs"
OUT = ROOT / "work" / "logit_lens.json"

SR, CLIP_S = 16000, 10.0
SNR = -10
N_UTTS = 3
N_TRACKS = 20          # highest-leak tracks -- where there is something to suppress
SEED = 20260811


def rms(x):
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def mix(s, i, snr):
    n = min(len(s), len(i))
    s, i = s[:n].copy(), i[:n].copy()
    i *= (rms(s) / (10 ** (snr / 20.0))) / (rms(i) + 1e-12)
    y = s + i
    pk = np.abs(y).max()
    return (y * (0.99 / pk) if pk > 0.99 else y).astype(np.float32)


def loudest(x, sec=CLIP_S):
    w = int(sec * SR)
    if len(x) <= w:
        return 0
    best, be = 0, -1.0
    for s0 in range(0, len(x) - w, SR // 2):
        e = float(np.sum(x[s0:s0 + w] ** 2))
        if e > be:
            be, best = e, s0
    return best


# --- pick the tracks where CTC leaked most: those have decodable lyrics ---
ctc = json.loads((RUNS / "ctc_ts.json").read_text())
solo = {}
for r in ctc:
    if r["cond"] == "vocals_alone" and r["track"]:
        solo[r["track"]] = r["hyp"]
qw = json.loads((RUNS / "qwen3_asr_1.7b_ts.json").read_text())
for r in qw:                       # Qwen transcribes singing best -> better lyric text
    if r["cond"] == "vocals_alone" and r["track"] and len(r["hyp"]) > len(solo.get(r["track"], "")):
        solo[r["track"]] = r["hyp"]
tracks = sorted(solo, key=lambda t: -len(solo[t]))[:N_TRACKS]
print(f"{len(tracks)} tracks with the most decodable lyrics", flush=True)

random.seed(SEED)
cands = [f for f in sorted(DATA.glob("LibriSpeech/test-clean/*/*/*.flac"))
         if 8.0 <= sf.info(str(f)).duration <= 14.0]
random.shuffle(cands)
utts = cands[:N_UTTS]

trans = {}
for tf in DATA.glob("LibriSpeech/test-clean/*/*/*.trans.txt"):
    for line in tf.read_text().splitlines():
        uid, text = line.split(" ", 1)
        trans[uid] = text

proc = WhisperProcessor.from_pretrained(str(MODEL))
model = WhisperForConditionalGeneration.from_pretrained(
    str(MODEL), torch_dtype=torch.float16).to("cuda").eval()
tok = proc.tokenizer
ln_f = model.model.decoder.layer_norm
head = model.proj_out


def token_ids(text):
    """content-word token ids, lowercase and capitalised variants"""
    ids = set()
    for w in text.split():
        w = w.strip(".,!?;:\"'").lower()
        if len(w) < 3:
            continue
        for form in (" " + w, " " + w.capitalize()):
            enc = tok.encode(form, add_special_tokens=False)
            if enc:
                ids.add(enc[0])
    return ids


rows = []


@torch.no_grad()
def run(audio, lyric_ids, speech_ids):
    feats = proc(audio, sampling_rate=SR, return_tensors="pt").input_features
    feats = feats.to("cuda", torch.float16)
    enc = model.model.encoder(feats)
    ids = model.generate(feats, max_new_tokens=180)
    dec = model.model.decoder(input_ids=ids,
                              encoder_hidden_states=enc.last_hidden_state,
                              output_hidden_states=True)
    ly = torch.tensor(sorted(lyric_ids), device="cuda")
    sp = torch.tensor(sorted(speech_ids), device="cuda")
    per_layer = []
    for h in dec.hidden_states:                       # 33 = embeddings + 32
        logits = head(ln_f(h))[0].float()             # (T, V)
        p = torch.softmax(logits, dim=-1)
        per_layer.append((float(p[:, ly].sum(-1).mean()),
                          float(p[:, sp].sum(-1).mean())))
    return per_layer, proc.batch_decode(ids, skip_special_tokens=True)[0].strip()


for ti, tr in enumerate(tracks, 1):
    x, _ = sf.read(str(STEMS / tr / "vocals.wav"), dtype="float32")
    seg = x[loudest(x):][:int(CLIP_S * SR)]
    if len(seg) < int(CLIP_S * SR) * 0.9:
        continue
    lyric_words = set(solo[tr].split())
    for flac in utts:
        sp_audio, _ = sf.read(str(flac), dtype="float32")
        ref = trans[flac.stem]
        l_ids = token_ids(solo[tr])
        s_ids = token_ids(ref)
        l_only, s_only = l_ids - s_ids, s_ids - l_ids
        if not l_only or not s_only:
            continue
        pl, hyp = run(mix(sp_audio, seg, SNR), l_only, s_only)
        rows.append({"track": tr, "utt": flac.stem, "hyp": hyp,
                     "p_lyric": [a for a, _ in pl], "p_speech": [b for _, b in pl]})
    if ti % 5 == 0:
        print(f"  [{ti}/{len(tracks)}] {len(rows)} clips", flush=True)

OUT.write_text(json.dumps(rows, indent=1))
print(f"wrote {OUT} ({len(rows)} clips)")

L = len(rows[0]["p_lyric"])
print("\nlayer   p_lyric   p_speech   ratio")
for i in range(L):
    a = float(np.mean([r["p_lyric"][i] for r in rows]))
    b = float(np.mean([r["p_speech"][i] for r in rows]))
    tag = "emb" if i == 0 else f"L{i:02d}"
    print(f"  {tag}  {a:.5f}   {b:.5f}   {a/(b+1e-9):7.3f}")
