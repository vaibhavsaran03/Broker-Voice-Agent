"""Data models for the broker-verification voice agent.

Everything here is plain pydantic so the live-call tools, the post-call
LangGraph graph, and the frontend all share one vocabulary.
"""
from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Furnishing(str, Enum):
    UNFURNISHED = "unfurnished"
    SEMI = "semi-furnished"
    FULLY = "fully-furnished"
    UNKNOWN = "unknown"


class VerificationStatus(str, Enum):
    UNVERIFIED = "unverified"
    VERIFIED_AVAILABLE = "verified_available"
    VERIFIED_UNAVAILABLE = "verified_unavailable"  # sold / already rented out
    STALE = "stale"  # broker could not confirm
    CONFLICT = "conflict"  # broker facts contradict the listing


class Listing(BaseModel):
    """A listing as scraped/received from a source. Mock data in this prototype."""

    id: str
    source: str  # e.g. "99acres", "magicbricks", "nobroker", "broker-whatsapp"
    broker_name: str
    broker_phone: str
    society: str
    address: str
    sector: str
    city: str
    bhk: int
    rent: int  # INR / month as advertised
    deposit: Optional[int] = None
    available_from: Optional[date] = None
    furnishing: Furnishing = Furnishing.UNKNOWN
    listed_at: datetime
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    canonical_property_id: Optional[str] = None  # set by entity resolution


class CallFacts(BaseModel):
    """Structured facts the voice agent pins down during a broker call."""

    is_available: Optional[bool] = None
    rent: Optional[int] = None
    deposit: Optional[int] = None
    available_from: Optional[date] = None
    furnishing: Optional[Furnishing] = None
    visit_slots: list[str] = Field(default_factory=list)
    brokerage_fee: Optional[str] = None
    notes: list[str] = Field(default_factory=list)


class TurnLatency(BaseModel):
    """One conversation turn's measured timing, in milliseconds."""

    turn: int
    user_speech_ms: Optional[float] = None  # how long the user spoke
    stt_ms: Optional[float] = None  # user stopped speaking -> final transcript
    llm_ms: Optional[float] = None  # final transcript -> first LLM token
    tts_ms: Optional[float] = None  # first LLM token -> first audio byte
    total_response_ms: Optional[float] = None  # user stopped -> agent audio starts


class CallRecord(BaseModel):
    id: str
    listing_id: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    transcript: list[dict] = Field(default_factory=list)  # {role, text, ts}
    facts: CallFacts = Field(default_factory=CallFacts)
    latency: list[TurnLatency] = Field(default_factory=list)
    verification_status: Optional[VerificationStatus] = None
    duplicate_of: Optional[str] = None  # canonical_property_id if flagged
    summary: Optional[str] = None


class DuplicateCandidate(BaseModel):
    listing_id: str
    canonical_property_id: str
    score: float  # 0..1
    reasons: list[str]
