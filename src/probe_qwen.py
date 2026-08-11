import base64
import json
import urllib.request
from pathlib import Path

URL = "http://[fdbd:dc55:d:1400::23]:9705/v1/transcribe"
clip = sorted(Path("/opt/tiger/icassp2027/work/clips").glob("*speech_alone*"))[0]
b64 = base64.b64encode(clip.read_bytes()).decode()
print("clip:", clip.name)

for payload in ({"audio": b64}, {"audio": "data:audio/wav;base64," + b64},
                {"audio_base64": b64}, {"file": b64}):
    key = list(payload)[0] + ("(data-uri)" if payload[list(payload)[0]].startswith("data:") else "")
    req = urllib.request.Request(
        URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            body = r.read().decode()
        print(f"  {key}: HTTP {r.status}")
        print("   ", body[:400])
        break
    except Exception as e:
        detail = ""
        if hasattr(e, "read"):
            try:
                detail = e.read().decode()[:250]
            except Exception:
                pass
        print(f"  {key}: {type(e).__name__} {e} {detail}")
