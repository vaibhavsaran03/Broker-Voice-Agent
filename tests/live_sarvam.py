"""Live Sarvam round-trip check (needs SARVAM_API_KEY in .env or the env).

Synthesizes a Hinglish sentence with Bulbul v3 and transcribes it back with
Saaras v3 in codemix mode, printing measured REST round-trip times. Not part
of the keyless pytest suite - run directly:  python tests/live_sarvam.py
"""
import base64
import json
import os
import time

import requests

KEY = os.environ.get("SARVAM_API_KEY") or open(
    os.path.join(os.path.dirname(__file__), "..", ".env")
).read().split("=", 1)[1].strip()
H = {"api-subscription-key": KEY}
SENT = ("Haan ji, flat available hai. Rent thirty five thousand, "
        "deposit two months. Visit Saturday morning possible hai.")

t0 = time.perf_counter()
r = requests.post("https://api.sarvam.ai/text-to-speech", headers=H, json={
    "text": SENT, "target_language_code": "hi-IN", "model": "bulbul:v3",
    "speaker": "neha", "speech_sample_rate": 16000}, timeout=60)
tts_ms = (time.perf_counter() - t0) * 1000
r.raise_for_status()
audio = base64.b64decode(r.json()["audios"][0])

t0 = time.perf_counter()
import io
r = requests.post("https://api.sarvam.ai/speech-to-text", headers=H,
    files={"file": ("speech.wav", io.BytesIO(audio), "audio/wav")},
    data={"model": "saaras:v3", "mode": "codemix", "language_code": "unknown"},
    timeout=60)
stt_ms = (time.perf_counter() - t0) * 1000
r.raise_for_status()

print(json.dumps({"tts_ms": round(tts_ms, 1), "stt_ms": round(stt_ms, 1),
                  "transcript": r.json().get("transcript")}, indent=2, ensure_ascii=False))
