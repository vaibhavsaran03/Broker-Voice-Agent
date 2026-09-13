"""The live voice pipeline (Pipecat).

Browser mic -> Silero VAD -> Sarvam Saaras (STT, Hinglish codemix mode)
-> Groq LLM with function calling -> Sarvam Bulbul (TTS) -> browser speaker.

The LLM drives the verification call: it asks one question at a time and
records facts through tools instead of free-writing them, so the structured
record only ever contains what the broker actually said.
"""
from __future__ import annotations

import asyncio
import io
import re
import time
import wave
from datetime import date, datetime
from typing import Any, Awaitable, Callable

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
    Frame,
    MetricsFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import TTFBMetricsData
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.services.sarvam.tts import SarvamTTSService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

from . import db
from .schemas import CallRecord, Furnishing, Listing, TurnLatency

EventSender = Callable[[dict], Awaitable[None]]

SYSTEM_PROMPT = """You are Riya, an assistant at a rental marketplace. You are ON A PHONE CALL
with a broker to verify one specific listing before showing it to a tenant.

Listing you are verifying:
- {bhk}BHK in {society}, {address}, {sector}, {city}
- Advertised rent: Rs {rent}/month, deposit Rs {deposit}

Rules of the call:
1. Introduce yourself in one short line and say which property you are calling about.
2. Your goals, in order: (a) is the flat still available, (b) exact rent, (c) deposit,
   (d) available-from date, (e) visit slots, (f) brokerage fee.
3. Ask ONE question at a time. Keep every reply under two short sentences - this is a
   phone call, not an email.
4. Brokers are often vague or switch between Hindi and English. Stay polite, pin down
   exact numbers, and repeat numbers back to confirm them.
5. The moment the broker states a fact, record it with the record_fact tool. Never
   invent a value the broker did not state. If the broker contradicts the advertised
   rent, record the broker's number and note the conflict.
6. When you have everything or the broker cannot confirm availability, thank them and
   end the call with the end_call tool.
7. The broker may speak Hindi or Hinglish; you may reply in simple English or Hinglish,
   matching them. Keep it natural.
"""


class WavSarvamSTTService(SarvamSTTService):
    """Fix two silent failures found by tests/live_pipeline.py (connection
    OK, zero transcripts back):

    1. Sarvam's streaming SDK only accepts encoding='audio/wav', but pipecat
       hands the service raw PCM frames - so wrap each chunk in a WAV header.
    2. The server ignores sub-100ms WAV messages without any error, and
       WebRTC audio arrives in 20ms frames - so buffer PCM and send ~200ms
       chunks, flushing the remainder when the user stops speaking.
    """

    MIN_CHUNK_BYTES = 6400  # 200 ms of 16 kHz mono PCM16

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pcm_buf = bytearray()

    def _wrap_wav(self, pcm: bytes) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate or 16000)
            w.writeframes(pcm)
        return buf.getvalue()

    async def _send_pcm(self, pcm: bytes):
        async for _frame in super().run_stt(self._wrap_wav(pcm)):
            pass

    async def run_stt(self, audio: bytes):
        self._pcm_buf += audio
        while len(self._pcm_buf) >= self.MIN_CHUNK_BYTES:
            chunk = bytes(self._pcm_buf[: self.MIN_CHUNK_BYTES])
            del self._pcm_buf[: self.MIN_CHUNK_BYTES]
            await self._send_pcm(chunk)
        yield None

    async def _connect(self):
        """Own the connection lifecycle. pipecat 0.0.108's _connect wires the
        receive task through its task manager in a way that leaves the socket
        deaf (sends land, no messages ever come back - proven by replaying the
        captured wire bytes over a clean connection in tests). Plain asyncio
        tasks work; verified live."""
        from sarvamai import AsyncSarvamAI
        from sarvamai.core.events import EventType

        client = AsyncSarvamAI(api_subscription_key=self._api_key)
        self._ws_cm = client.speech_to_text_streaming.connect(
            model=self._settings.model,
            mode=self._mode,
            language_code=self._get_language_string() or "unknown",
            sample_rate=str(self.sample_rate or 16000),
            flush_signal="true",
        )
        self._socket_client = await self._ws_cm.__aenter__()
        self._socket_client.on(
            EventType.MESSAGE,
            lambda m: asyncio.create_task(self._handle_message(m)),
        )
        self._receive_task = asyncio.create_task(self._receive_task_handler())

    async def _disconnect(self):
        if self._receive_task:
            self._receive_task.cancel()
            self._receive_task = None
        self._socket_client = None
        cm = getattr(self, "_ws_cm", None)
        if cm:
            try:
                await cm.__aexit__(None, None, None)
            except Exception:
                pass
            self._ws_cm = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, (UserStoppedSpeakingFrame, VADUserStoppedSpeakingFrame)):
            await asyncio.sleep(0.2)  # let queued audio generators drain into the buffer
            if self._pcm_buf:
                chunk = bytes(self._pcm_buf)
                self._pcm_buf.clear()
                await self._send_pcm(chunk)
        await super().process_frame(frame, direction)


# Tool schema shared by the live transport pipeline and the in-process
# live test (tests/live_pipeline.py).
TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "record_fact",
                "description": "Record one fact stated by the broker. Call this the moment the broker states it.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "field": {
                            "type": "string",
                            "enum": ["is_available", "rent", "deposit", "available_from",
                                     "furnishing", "visit_slot", "brokerage_fee", "note"],
                        },
                        "value": {"type": "string",
                                  "description": "rent/deposit as digits or '35k'; available_from as YYYY-MM-DD"},
                    },
                    "required": ["field", "value"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "end_call",
                "description": "End the call once all facts are captured or the flat is confirmed unavailable.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
]


class LatencyTracker(FrameProcessor):
    """Sits at the end of the pipeline and measures every turn."""

    def __init__(self, record: CallRecord, send: EventSender):
        super().__init__()
        self._record = record
        self._send = send
        self._user_started: float | None = None
        self._user_stopped: float | None = None
        self._turn = 0
        self._pending = TurnLatency(turn=0)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        now = time.monotonic()

        if isinstance(frame, UserStartedSpeakingFrame):
            self._user_started = now
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._user_stopped = now
            self._turn += 1
            self._pending = TurnLatency(turn=self._turn)
            if self._user_started:
                self._pending.user_speech_ms = round((now - self._user_started) * 1000, 1)
        elif isinstance(frame, TranscriptionFrame) and self._user_stopped:
            if self._pending.stt_ms is None:
                self._pending.stt_ms = round((now - self._user_stopped) * 1000, 1)
        elif isinstance(frame, MetricsFrame):
            for m in frame.data:
                if isinstance(m, TTFBMetricsData) and m.value:
                    ms = round(m.value * 1000, 1)
                    if m.processor.lower().startswith(("sarvamstt", "stt")):
                        pass  # STT timing comes from the transcript frame above
                    elif m.processor.lower().startswith(("groq", "llm")):
                        if self._pending.llm_ms is None:
                            self._pending.llm_ms = ms
                    elif m.processor.lower().startswith(("sarvamtts", "tts")):
                        if self._pending.tts_ms is None:
                            self._pending.tts_ms = ms
        elif isinstance(frame, BotStartedSpeakingFrame) and self._user_stopped:
            if self._pending.total_response_ms is None:
                self._pending.total_response_ms = round((now - self._user_stopped) * 1000, 1)
                self._record.latency.append(self._pending)
                db.save_call(self._record)
                await self._send({"type": "latency", "turn": self._pending.model_dump()})
                self._user_stopped = None

        await self.push_frame(frame, direction)


def _parse_fact(record: CallRecord, field: str, value: str) -> str:
    """Store one broker-stated fact. Returns a short ack string for the LLM."""
    f = record.facts
    v = value.strip()
    low = v.lower()
    try:
        if field == "is_available":
            f.is_available = low not in ("no", "false", "not available", "unavailable")
        elif field in ("rent", "deposit"):
            num = re.search(r"[\d,]+(?:\.\d+)?k?", low)
            if num is None:
                raise ValueError("no number found")
            val = float(num.group(0).replace(",", "").replace("k", "000"))
            if val < 100:
                # "2 months" is not Rs 2 - refuse to guess; the post-call graph
                # derives month-based deposits from an explicit note instead.
                raise ValueError("implausibly small amount")
            if field == "rent":
                f.rent = int(val)
            else:
                f.deposit = int(val)
        elif field == "available_from":
            f.available_from = date.fromisoformat(v)
        elif field == "furnishing":
            f.furnishing = Furnishing(low) if low in Furnishing._value2member_map_ else Furnishing.UNKNOWN
        elif field == "visit_slot":
            f.visit_slots.append(v)
        elif field == "brokerage_fee":
            f.brokerage_fee = v
        elif field == "note":
            f.notes.append(v)
        else:
            return f"unknown field {field}"
    except (ValueError, IndexError):
        return f"could not parse {field}={value!r}"
    db.save_call(record)
    return "recorded"


async def run_agent(
    webrtc_connection,
    listing: Listing,
    record: CallRecord,
    sarvam_api_key: str,
    groq_api_key: str,
    groq_model: str = "openai/gpt-oss-20b",
    sarvam_stt_model: str = "saaras:v3",
    sarvam_tts_model: str = "bulbul:v3",
    sarvam_tts_voice: str = "neha",
) -> None:
    """Run one broker-verification call until the browser hangs up."""

    async def send(message: dict) -> None:
        try:
            webrtc_connection.send_app_message(message)
        except Exception:
            pass  # browser already gone; call record still holds everything

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )
    stt = WavSarvamSTTService(
        api_key=sarvam_api_key,
        model=sarvam_stt_model,
        params=SarvamSTTService.InputParams(mode="codemix"),
    )
    llm = GroqLLMService(
        api_key=groq_api_key,
        model=groq_model,
        # gpt-oss on Groq defaults to medium reasoning: it loops tool calls and
        # never speaks (and invented a fact in testing). Low effort talks.
        params=GroqLLMService.InputParams(extra={"reasoning_effort": "low"}),
    )
    tts = SarvamTTSService(
        api_key=sarvam_api_key, model=sarvam_tts_model, voice_id=sarvam_tts_voice
    )

    async def record_fact_handler(params):
        args = params.arguments
        ack = _parse_fact(record, args.get("field", ""), str(args.get("value", "")))
        await send({"type": "facts", "facts": record.facts.model_dump(mode="json")})
        await params.result_callback(ack)

    async def end_call_handler(params):
        await send({"type": "status", "status": "call_ended_by_agent"})
        await params.result_callback("Call ended. Thank the broker and say goodbye.")
        await task.queue_frames([_end_frame()])

    llm.register_function("record_fact", record_fact_handler)
    llm.register_function("end_call", end_call_handler)

    tools = TOOLS

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(
            bhk=listing.bhk, society=listing.society, address=listing.address,
            sector=listing.sector, city=listing.city, rent=f"{listing.rent:,}",
            deposit=f"{listing.deposit:,}" if listing.deposit else "not advertised",
        )},
        {"role": "user", "content": "The broker has picked up the call. Greet them and begin."},
    ]
    context = OpenAILLMContext(messages=messages, tools=tools)
    aggregators = llm.create_context_aggregator(context)

    class TranscriptRelay(FrameProcessor):
        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)
            if isinstance(frame, TranscriptionFrame):
                record.transcript.append(
                    {"role": "broker", "text": frame.text, "ts": datetime.utcnow().isoformat()}
                )
                await send({"type": "transcript", "role": "broker", "text": frame.text})
            await self.push_frame(frame, direction)

    tracker = LatencyTracker(record, send)
    pipeline = Pipeline([
        transport.input(),
        stt,
        TranscriptRelay(),
        aggregators.user(),
        llm,
        tts,
        transport.output(),
        aggregators.assistant(),
        tracker,
    ])
    task = PipelineTask(pipeline, params=PipelineParams(
        enable_metrics=True, enable_usage_metrics=True, allow_interruptions=True,
    ))
    runner = PipelineRunner()

    @transport.event_handler("on_client_disconnected")
    async def _on_disconnect(_transport, _connection):
        await task.cancel()

    await send({"type": "status", "status": "connected", "call_id": record.id})
    await runner.run(task)
    record.ended_at = datetime.utcnow()
    db.save_call(record)


def _end_frame():
    from pipecat.frames.frames import EndFrame
    return EndFrame()
