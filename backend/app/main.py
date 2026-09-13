"""FastAPI entry point.

Serves the single-page demo, the REST surface for listings/calls, and the
WebRTC offer endpoint the browser connects to for the live voice call.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pipecat.transports.smallwebrtc.connection import IceServer, SmallWebRTCConnection

from . import db, graph

load_dotenv()

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="Broker-Voice-Agent (prototype)")


@app.on_event("startup")
def _startup() -> None:
    db.init_db()
    db.seed_mock_listings()


def _xirsys_ice_config() -> list[dict]:
    """Mint fresh provider credentials for each page/offer (max TTL 6 hours)."""
    ident = os.getenv("XIRSYS_IDENT", "")
    secret = os.getenv("XIRSYS_SECRET", "")
    channel = os.getenv("XIRSYS_CHANNEL", "")
    if ident and secret and channel:
        auth = base64.b64encode(f"{ident}:{secret}".encode()).decode()
        request = urllib.request.Request(
            f"https://global.xirsys.net/_turn/{channel}?webrtc=1&expire=21600",
            method="PUT",
            headers={"Authorization": f"Basic {auth}"},
        )
        with urllib.request.urlopen(request, timeout=8) as response:
            payload = json.load(response)
        if payload.get("s") != "ok":
            import logging
            logging.getLogger(__name__).error("Xirsys credential request failed: status=%r value_type=%s", payload.get("s"), type(payload.get("v")).__name__)
            raise RuntimeError("Xirsys did not return TURN credentials")
        return payload["v"]["iceServers"] if isinstance(payload.get("v"), dict) else json.loads(payload["v"])["iceServers"]

    # Local development can run STUN-only. Production config uses Xirsys.
    return [{"urls": os.getenv("STUN_URL", "stun:stun.l.google.com:19302")}]


def _ice_servers(config: list[dict]) -> list:
    servers = []
    for item in config:
        servers.append(IceServer(
            urls=item["urls"],
            username=item.get("username"),
            credential=item.get("credential"),
        ))
    return servers


@app.get("/api/ice-servers")
def api_ice_servers():
    try:
        return _xirsys_ice_config()
    except Exception as exc:
        raise HTTPException(503, f"TURN credential service unavailable: {type(exc).__name__}")


class Offer(BaseModel):
    sdp: str
    type: str
    call_id: str


@app.get("/api/listings")
def api_listings():
    return [l.model_dump(mode="json") for l in db.list_listings()]


@app.post("/api/calls")
def api_new_call(listing_id: str):
    if db.get_listing(listing_id) is None:
        raise HTTPException(404, "unknown listing")
    return db.new_call(listing_id).model_dump(mode="json")


@app.get("/api/calls")
def api_calls():
    return [c.model_dump(mode="json") for c in db.list_calls()]


@app.get("/api/calls/{call_id}")
def api_call(call_id: str):
    record = db.get_call(call_id)
    if record is None:
        raise HTTPException(404, "unknown call")
    return record.model_dump(mode="json")


@app.post("/api/calls/{call_id}/process")
def api_process(call_id: str):
    try:
        state = graph.run_post_call(call_id)
    except ValueError:
        raise HTTPException(404, "unknown call")
    record = state["record"]
    return {
        "verification_status": record.verification_status,
        "duplicate_of": state.get("duplicate_of"),
        "duplicate_reasons": state.get("duplicate_reasons", []),
        "summary": record.summary,
    }


@app.post("/api/offer")
async def api_offer(offer: Offer):
    """Browser POSTs its SDP offer here; we answer and start the voice agent."""
    sarvam_key = os.getenv("SARVAM_API_KEY", "")
    groq_key = os.getenv("GROQ_API_KEY", "")
    if not sarvam_key or not groq_key:
        raise HTTPException(
            503,
            "voice pipeline needs SARVAM_API_KEY and GROQ_API_KEY in the environment "
            "(see .env.example). Everything else in this app runs without keys.",
        )

    record = db.get_call(offer.call_id)
    listing = db.get_listing(record.listing_id) if record else None
    if record is None or listing is None:
        raise HTTPException(404, "unknown call_id - POST /api/calls first")

    from .pipeline import run_agent  # deferred: keeps keyless endpoints importable

    try:
        ice_config = await asyncio.to_thread(_xirsys_ice_config)
    except Exception as exc:
        raise HTTPException(503, f"TURN credential service unavailable: {type(exc).__name__}")
    connection = SmallWebRTCConnection(ice_servers=_ice_servers(ice_config))
    await connection.initialize(sdp=offer.sdp, type=offer.type)
    answer = connection.get_answer()
    if answer is None:
        raise HTTPException(400, "could not build SDP answer")

    async def _run():
        try:
            await run_agent(
                connection,
                listing,
                record,
                sarvam_api_key=sarvam_key,
                groq_api_key=groq_key,
                groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
                sarvam_stt_model=os.getenv("SARVAM_STT_MODEL", "saaras:v3"),
                sarvam_tts_model=os.getenv("SARVAM_TTS_MODEL", "bulbul:v3"),
                sarvam_tts_voice=os.getenv("SARVAM_TTS_VOICE", "neha"),
            )
        finally:
            # whatever ended the call, the post-call graph runs on what was captured
            try:
                graph.run_post_call(record.id)
            except Exception:
                pass
            await connection.cleanup()

    asyncio.create_task(_run())
    return answer


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
