"""Live end-to-end pipeline test (needs SARVAM_API_KEY and GROQ_API_KEY in .env).

No browser, no WebRTC: a scripted source processor plays a Bulbul-synthesized
Hinglish "broker" utterance into the real pipeline (Sarvam Saaras STT in
codemix mode -> Groq LLM with the production system prompt + tools -> Sarvam
Bulbul TTS), and the production LatencyTracker measures the turn.

Run directly:  python tests/live_pipeline.py
Not part of the keyless pytest suite.
"""
import asyncio
import base64
import io
import json
import os
import sys
import time
import wave
from datetime import datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipecat.frames.frames import (  # noqa: E402
    BotStartedSpeakingFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    LLMTextFrame,
    StartFrame,
    TTSStartedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline  # noqa: E402
from pipecat.pipeline.runner import PipelineRunner  # noqa: E402
from pipecat.pipeline.task import PipelineParams, PipelineTask  # noqa: E402
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext  # noqa: E402
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor  # noqa: E402
from pipecat.services.groq.llm import GroqLLMService  # noqa: E402
from pipecat.services.sarvam.stt import SarvamSTTService  # noqa: E402
from pipecat.services.sarvam.tts import SarvamTTSService  # noqa: E402

from backend.app import db  # noqa: E402
from backend.app.pipeline import SYSTEM_PROMPT, LatencyTracker, WavSarvamSTTService, _parse_fact  # noqa: E402
from backend.app.schemas import CallRecord, Listing  # noqa: E402

# ------------------------------------------------------------------ env
env_path = Path(__file__).resolve().parent.parent / ".env"
for line in env_path.read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

SARVAM_KEY = os.environ["SARVAM_API_KEY"]
GROQ_KEY = os.environ["GROQ_API_KEY"]

BROKER_LINE = ("Haan ji, flat available hai. Rent thirty five thousand hai, "
               "deposit two months. Visit Saturday morning possible hai.")


def synthesize_broker_audio() -> bytes:
    """Bulbul v3 -> 16 kHz mono PCM16 bytes of the broker line."""
    r = requests.post(
        "https://api.sarvam.ai/text-to-speech",
        headers={"api-subscription-key": SARVAM_KEY},
        json={"text": BROKER_LINE, "target_language_code": "hi-IN",
              "model": "bulbul:v3", "speaker": "neha", "speech_sample_rate": 16000},
        timeout=60)
    r.raise_for_status()
    wav = base64.b64decode(r.json()["audios"][0])
    with wave.open(io.BytesIO(wav)) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        return w.readframes(w.getnframes())


class ScriptedSource(FrameProcessor):
    """Plays pre-recorded audio downstream as if it were a live mic."""

    def __init__(self, pcm: bytes):
        super().__init__()
        self._pcm = pcm

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            asyncio.create_task(self._play())
        await self.push_frame(frame, direction)

    async def _play(self):
        try:
            await self._play_inner()
        except Exception:
            import traceback; traceback.print_exc()

    async def _play_inner(self):
        await asyncio.sleep(1.0)  # let services connect
        await self.push_frame(UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        chunk = 640  # 20 ms of 16 kHz mono PCM16
        for i in range(0, len(self._pcm), chunk):
            await self.push_frame(
                InputAudioRawFrame(audio=self._pcm[i:i + chunk],
                                   sample_rate=16000, num_channels=1),
                FrameDirection.DOWNSTREAM)
            await asyncio.sleep(0.02)  # real-time pace
        await asyncio.sleep(0.3)
        await self.push_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)


class Sink(FrameProcessor):
    """Terminal processor: synthesizes the bot-speech marker the transport
    would normally emit, and collects observability."""

    def __init__(self):
        super().__init__()
        self.agent_text: list[str] = []
        self.tts_audio_bytes = 0
        self._bot_marker_sent = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSStartedFrame) and not self._bot_marker_sent:
            # SmallWebRTCTransport would emit this; no transport here.
            await self.push_frame(BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
            self._bot_marker_sent = True
        elif isinstance(frame, LLMTextFrame):
            self.agent_text.append(frame.text)
        from pipecat.frames.frames import TTSAudioRawFrame
        if isinstance(frame, TTSAudioRawFrame):
            self.tts_audio_bytes += len(frame.audio)
        await self.push_frame(frame, direction)


async def main() -> None:

    db.init_db(Path("/tmp/live_pipeline_test.db"))
    listing = Listing(
        id="test-1", source="99acres", broker_name="Test Broker",
        broker_phone="+919999999999", society="Sobha Dream Acres",
        address="Tower 4, Flat 1203", sector="Varthur", city="Bengaluru",
        bhk=2, rent=35000, deposit=70000, listed_at=datetime.utcnow())
    record = CallRecord(id="live-test-1", listing_id=listing.id,
                        started_at=datetime.utcnow())

    done = asyncio.Event()
    events: list[dict] = []

    async def send(message: dict) -> None:
        events.append(message)
        if message.get("type") == "latency":
            done.set()

    stt = WavSarvamSTTService(api_key=SARVAM_KEY, model="saaras:v3",
                              params=SarvamSTTService.InputParams(mode="codemix"))
    llm = GroqLLMService(api_key=GROQ_KEY, model="openai/gpt-oss-20b",
                          params=GroqLLMService.InputParams(extra={"reasoning_effort": "low"}))
    tts = SarvamTTSService(api_key=SARVAM_KEY, model="bulbul:v3", voice_id="neha")

    async def record_fact_handler(params):
        args = params.arguments
        ack = _parse_fact(record, args.get("field", ""), str(args.get("value", "")))
        await send({"type": "facts", "facts": record.facts.model_dump(mode="json")})
        await params.result_callback(ack)

    async def end_call_handler(params):
        await params.result_callback("Call ended.")

    llm.register_function("record_fact", record_fact_handler)
    llm.register_function("end_call", end_call_handler)

    from backend.app.pipeline import TOOLS  # production tool schema
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(
            bhk=listing.bhk, society=listing.society, address=listing.address,
            sector=listing.sector, city=listing.city, rent=f"{listing.rent:,}",
            deposit=f"{listing.deposit:,}" if listing.deposit else "not advertised")},
        {"role": "user", "content": "The broker has picked up the call. Greet them and begin."},
    ]
    context = OpenAILLMContext(messages=messages, tools=TOOLS)
    aggregators = llm.create_context_aggregator(context)

    pcm = synthesize_broker_audio()
    source = ScriptedSource(pcm)
    sink = Sink()
    tracker = LatencyTracker(record, send)

    pipeline = Pipeline([
        source, stt, aggregators.user(), llm, tts,
        aggregators.assistant(), sink, tracker,
    ])
    task = PipelineTask(pipeline, params=PipelineParams(
        enable_metrics=True, enable_usage_metrics=True, allow_interruptions=True))
    runner = PipelineRunner()

    async def watchdog():
        try:
            await asyncio.wait_for(done.wait(), timeout=150)
        except asyncio.TimeoutError:
            pass
        await asyncio.sleep(2)  # let TTS audio flush
        await task.queue_frames([EndFrame()])

    wd = asyncio.create_task(watchdog())
    t0 = time.perf_counter()
    try:
        await asyncio.wait_for(runner.run(task), timeout=170)
    except asyncio.TimeoutError:
        print("[main] runner timed out, printing partial result", flush=True)
    total_s = time.perf_counter() - t0
    wd.cancel()

    result = {
        "broker_line_in": BROKER_LINE,
        "stt_transcript": [t["text"] for t in record.transcript],
        "agent_reply": "".join(sink.agent_text)[:600],
        "facts": record.facts.model_dump(mode="json"),
        "latency_ms": [t.model_dump() for t in record.latency],
        "tts_audio_bytes": sink.tts_audio_bytes,
        "wall_time_s": round(total_s, 1),
        "events": [e.get("type") for e in events],
    }
    print("LIVE_RESULT_JSON " + json.dumps(result, default=str))


if __name__ == "__main__":
    asyncio.run(main())
