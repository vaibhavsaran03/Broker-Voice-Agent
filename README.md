# Broker-Voice-Agent

A voice agent that calls a broker to verify a rental listing: is the flat
actually available, what's the real rent, deposit, available-from date, and
when can someone visit. It speaks with the broker (Hindi, Hinglish, or English),
pins down exact numbers through a vague, evasive conversation, writes
structured facts to a listing database, and flags when the same flat appears
under three different brokers.

**This is a prototype.** All listings and broker details in the demo are
invented mock data. Nothing here touches a real phone line yet - the "call"
happens over WebRTC in your browser, where **you play the broker** and try to
be as difficult as a real one.

## Why this exists

Indian rental portals have a verification problem: listings go stale, brokers
quote different numbers than the ad, and the same flat gets listed by multiple
brokers with slightly different spellings and rents ("Sector 108" vs "Sec 108"
vs "108"). A rental marketplace lives or dies on knowing what's *actually*
available. This project is that verification loop as a working voice agent.

## Demo

```bash
cp .env.example .env   # add SARVAM_API_KEY and GROQ_API_KEY (both have free tiers)
docker compose up --build
# open http://localhost:8000
```

1. Pick a listing (try any **Sobha City** one - there are three copies of the
   same flat in the mock data, listed by different brokers).
2. Click *Start verification call*. The browser asks for your mic.
3. You are the broker now. Be vague. Mix Hindi and English. Quote a different
   rent than the ad. Watch the structured-facts panel fill in and the per-turn
   latency table update as you talk.
4. Hang up. The post-call graph runs: verification verdict, rent-conflict
   detection, and a duplicate warning if this flat is listed elsewhere.

## Architecture

```
browser mic ──WebRTC──> Silero VAD ──> Sarvam Saaras (STT, codemix mode)
                                        │
                                        v
                              Groq LLM (gpt-oss-20b)
                              function calling: record_fact / end_call
                                        │
                                        v
                              Sarvam Bulbul (TTS) ──WebRTC──> browser speaker

call ends ──> LangGraph post-call graph (deterministic, no LLM):
              normalize facts -> verify vs listing -> entity resolution
              -> persist -> rule-based summary
```

Design choices worth explaining:

- **Sarvam for STT/TTS.** Indian rental calls are code-mixed by default.
  Saaras runs in `codemix` mode and Bulbul handles Hinglish natively, so the
  agent works the way brokers actually talk instead of demo-English.
- **Facts go through tools, never through prose.** The LLM can only write to
  the listing record via `record_fact(field, value)`. The post-call graph is
  deliberately deterministic code - no generative step after the call - so a
  bad model turn can't silently rewrite the database. If the broker says
  "deposit is 2 months", the agent records a *note*, and the graph derives
  `2 x rent` explicitly rather than letting the model do arithmetic.
- **Entity resolution is explainable.** Society name, address overlap, sector,
  BHK, rent proximity and broker phone each contribute weighted points, and
  every duplicate flag lists its reasons (`society match`, `same sector (108)`,
  `rent within 10%`). No embeddings-only black box.
- **Latency is measured, not quoted.** The UI shows per-turn timing captured
  from the running pipeline: how long you spoke, STT completion, LLM
  time-to-first-token, TTS time-to-first-byte, and total response time
  (you stop talking -> agent starts replying). Vendor pages quote their own
  numbers; this shows what the assembled pipeline actually costs.

## Verified against the live Sarvam API (13 Sep 2026)

Speech round-trip test (code in `tests/live_sarvam.py`, needs `SARVAM_API_KEY`):
synthesized a Hinglish sentence with Bulbul v3, transcribed it back with
Saaras v3 in codemix mode, measured on a home connection:

| stage | result |
| --- | --- |
| TTS (bulbul:v3, full audio for a ~17-word Hinglish sentence) | 4.0s REST round-trip |
| STT (saaras:v3, codemix mode, 16kHz wav) | 1.9s REST round-trip |
| Transcript of "Rent thirty five thousand, deposit two months" mixed into Hindi | "Rent 35,000, deposit 2 months" - numbers and English spans come back exact |

These are blocking REST totals (worst case), not the streaming time-to-first-byte
the pipeline actually runs at; the per-turn streaming numbers are what the demo
UI measures live. Note: `bulbul:v2` was deprecated by Sarvam - this repo uses
`bulbul:v3` throughout.

## Stack

Python 3.11, Pipecat (realtime voice pipeline), Sarvam Saaras + Bulbul
(speech), Groq (dialogue LLM), LangGraph (post-call processing), FastAPI,
SQLite, vanilla-JS WebRTC frontend, Docker.

## Repo layout

```
backend/app/pipeline.py           live voice pipeline + latency tracking
backend/app/graph.py              LangGraph post-call verification graph
backend/app/entity_resolution.py  duplicate-listing detection
backend/app/db.py                 SQLite store + mock seed data
backend/app/main.py               FastAPI: REST + WebRTC offer endpoint
backend/static/index.html         the demo page
tests/                            13 tests, all keyless (pytest)
```

## Testing

```bash
pip install -r requirements.txt
pytest tests/
```

Everything except the live voice loop is covered by keyless tests: fact
parsing (including the "35k" and "Rs 34,000" cases and refusing to guess
"2 months" as Rs 2), entity resolution across the planted triplicate
listings, the full post-call graph, and the API lifecycle.

## Limitations / next steps

- **Browser WebRTC, not telephony.** A real deployment needs SIP (Exotel/
  Twilio) so the agent dials actual broker numbers. Pipecat supports both;
  it's a transport swap, not a rewrite.
- Mock listings; no portal ingestion yet.
- Single concurrent call, SQLite, no auth - it's a demo, not a service.
- Free-tier speech credits are finite; long demo sessions will burn them.
- Turn-taking uses Silero VAD defaults; noisy broker audio would want tuning.

## Live pipeline status (13 Sep 2026, latest)

`tests/live_pipeline.py` drives the real pipeline in-process (no browser):
a scripted source plays a Bulbul-synthesized Hinglish "broker" utterance into
Sarvam STT -> Groq LLM (production system prompt + tools) -> Sarvam TTS, with
the production LatencyTracker attached.

Verified live in this harness:

- Streaming STT (saaras:v3, codemix) transcribes the broker utterance; numbers
  and English spans come through ("35000 hai deposit 2 months").
- The Groq LLM (openai/gpt-oss-20b) called the production `record_fact` tool
  and the production parser stored: is_available=true, rent=35000,
  deposit=70000 - matching what the "broker" said.
- Two real integration bugs were found and fixed by this test (see
  `WavSarvamSTTService` in `backend/app/pipeline.py`): Sarvam's streaming SDK
  only accepts WAV-wrapped audio, and it silently ignores sub-100ms chunks
  while WebRTC delivers 20ms frames. Also: Groq retired
  llama-3.3-70b-versatile; the default model is now openai/gpt-oss-20b.

Known issues, not yet fixed (recorded here so nothing is overstated):

- Sarvam's streaming STT websocket intermittently returns zero transcripts
  with a clean connection and successful sends (3 consecutive silent runs on
  13 Sep 2026 after several successful ones; no error frames, server-side
  silence). The full spoken turn + per-turn latency numbers are blocked on
  this.

- Tool-use behavior is model- and settings-sensitive. Direct API tests
  (same prompt, same tools, same broker transcript) show openai/gpt-oss-20b
  with reasoning_effort=low does the right thing: one round of record_fact
  calls (rent/deposit/availability exactly as stated), then a natural spoken
  follow-up ("Okay. You said the rent is 35,000 rupees. Is that 35,000 per
  month?"), ~300-450ms per call. With default (medium) reasoning it loops
  tool calls and once invented an available_from date the broker never said.
  The pipeline pins reasoning_effort=low via Settings (the params= path is
  silently dropped by pipecat's GroqLLMService - found by dumping the actual
  request payloads in tests/live_pipeline.py). The fact parser also accepts
  broker-style month deposits ("2 months" -> 2 x recorded rent) instead of
  rejecting them, which was driving a tool-retry loop, and rejections now
  tell the model to ask the broker rather than guess.

## Live demo

https://broker-voice-agent.onrender.com (Render free tier, Docker, Oregon -
same setup as the Invoice Auditor demo). Free tier sleeps after ~15 min idle,
so the first load can take ~50s (cold start); after that it is instant. The
demo page, listings API, and per-turn latency table are all live. Note:
auto-deploy is off (public-repo services only redeploy manually), and the
live WebRTC call leg from Render's proxy is verified in-browser only up to
the offer endpoint - the mic call is best tried from a phone.
