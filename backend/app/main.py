"""FastAPI entry point.

Serves the single-page demo, the REST surface for listings/calls, and the
WebRTC offer endpoint the browser connects to for the live voice call.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
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


def _ice_servers() -> list:
    """ICE servers for the WebRTC leg.

    Host candidates alone only work when browser and server share a network
    (local docker-compose). Behind Render's proxy the container's host
    candidate is a private IP, so a TURN relay is required. Production values
    come from TURN_URL/TURN_USERNAME/TURN_CREDENTIAL; local development keeps
    STUN-only behavior when those variables are absent.
    """
    servers = [IceServer(urls=os.getenv("STUN_URL", "stun:stun.l.google.com:19302"))]
    turn_url = os.getenv("TURN_URL", "")
    turn_user = os.getenv("TURN_USERNAME", "")
    turn_cred = os.getenv("TURN_CREDENTIAL", "")
    if turn_url:
        urls = [u.strip() for u in turn_url.split(",") if u.strip()]
        # aiortc supports one TURN server and selects the first TURN URI it
        # encounters. Prefer TLS/TCP 443 on hosted deployments because Render
        # can block or NAT UDP allocations; give aiortc a singleton URI.
        selected = next((u for u in urls if u.startswith("turns:") and ":443" in u), urls[0])
        servers.append(
            IceServer(
                urls=selected,
                username=turn_user,
                credential=turn_cred,
            )
        )
    return servers


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

    ice_servers = _ice_servers()
    print("ICE_CONFIG_SHAPE", [(x.urls, bool(x.username), bool(x.credential)) for x in ice_servers], flush=True)
    connection = SmallWebRTCConnection(ice_servers=ice_servers)
    # SmallWebRTCConnection builds its RTCPeerConnection during __init__, but
    # some releases initialize it before retaining constructor ICE servers.
    # Rebuild once after the configured servers are present so aiortc receives
    # the TURN config that will gather the answer candidates.
    connection._initialize()
    # aiortc returns from setLocalDescription before TURN candidate gathering
    # has necessarily finished. With one-shot signaling, wait for gathering so
    # the SDP answer contains the server relay candidate rather than only its
    # unreachable Render container/private candidates.
    original_pc_set_local = connection._pc.setLocalDescription
    async def _set_local_and_gather(description):
        await original_pc_set_local(description)
        deadline = asyncio.get_running_loop().time() + 8
        while connection._pc.iceGatheringState != "complete" and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.05)
    connection._pc.setLocalDescription = _set_local_and_gather
    await connection.initialize(sdp=offer.sdp, type=offer.type)
    answer = connection.get_answer()
    if answer is not None:
        answer_sdp = answer.get("sdp", "") if isinstance(answer, dict) else answer.sdp
        print("ANSWER_CANDIDATES", [line for line in answer_sdp.splitlines() if line.startswith("a=candidate")], flush=True)
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
    # Keep the browser ICE configuration in the same environment-backed source
    # of truth as the server. TURN credentials are necessarily exposed to the
    # WebRTC client, but are short-lived/provider-scoped rather than API keys.
    html = (STATIC_DIR / "index.html").read_text()
    browser_ice = [
        {"urls": os.getenv("STUN_URL", "stun:stun.l.google.com:19302")},
    ]
    turn_urls = [u.strip() for u in os.getenv("TURN_URL", "").split(",") if u.strip()]
    if turn_urls:
        browser_ice.append({
            "urls": turn_urls,
            "username": os.getenv("TURN_USERNAME", ""),
            "credential": os.getenv("TURN_CREDENTIAL", ""),
        })
    import json
    return HTMLResponse(html.replace("__ICE_SERVERS__", json.dumps(browser_ice)))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
