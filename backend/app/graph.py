"""Post-call processing graph (LangGraph).

Runs after a broker call ends:
  normalize_facts   - tidy what the live call captured (units, dates)
  verify_listing    - compare broker's answers against the advertised listing
  resolve_entities  - check whether this flat is already known under another broker
  persist           - write verification status + canonical property links
  summarize         - short factual call summary (no LLM needed - rule-based so
                      it stays honest: it can only say what was actually captured)

The graph is deliberately deterministic. The only generative step in the whole
system is the live conversation itself; everything after the call is checkable
code, so a wrong model answer can never silently rewrite the listing DB.
"""
from __future__ import annotations

from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from . import db
from .entity_resolution import find_duplicates
from .schemas import CallFacts, CallRecord, VerificationStatus


class CallState(TypedDict):
    call_id: str
    record: Optional[CallRecord]
    duplicate_of: Optional[str]
    duplicate_reasons: list[str]


def _normalize_facts(state: CallState) -> dict:
    record = state["record"]
    assert record is not None
    facts = record.facts
    # deposit sanity: many brokers quote deposit as "2 months rent"
    if facts.deposit is None and facts.rent is not None:
        for note in facts.notes:
            if "month" in note.lower() and "deposit" in note.lower():
                digits = [int(s) for s in note.split() if s.isdigit()]
                if digits:
                    facts.deposit = digits[0] * facts.rent
                    break
    db.save_call(record)
    return {"record": record}


def _verify_listing(state: CallState) -> dict:
    record = state["record"]
    assert record is not None
    listing = db.get_listing(record.listing_id)
    facts = record.facts

    if facts.is_available is False:
        record.verification_status = VerificationStatus.VERIFIED_UNAVAILABLE
    elif facts.is_available is True:
        conflict = False
        if listing and facts.rent and abs(facts.rent - listing.rent) > 0.15 * listing.rent:
            conflict = True  # broker quoted >15% off the advertised rent
        record.verification_status = (
            VerificationStatus.CONFLICT if conflict else VerificationStatus.VERIFIED_AVAILABLE
        )
    else:
        record.verification_status = VerificationStatus.STALE
    db.save_call(record)
    return {"record": record}


def _resolve_entities(state: CallState) -> dict:
    record = state["record"]
    assert record is not None
    listing = db.get_listing(record.listing_id)
    if listing is None:
        return {"duplicate_of": None, "duplicate_reasons": []}

    hits = find_duplicates(listing, db.list_listings())
    if not hits:
        return {"duplicate_of": None, "duplicate_reasons": []}

    best, result = hits[0]
    canonical = best.canonical_property_id or f"prop-{best.id}"
    # link every known copy of this flat to one canonical property id
    listing.canonical_property_id = canonical
    db.upsert_listing(listing)
    best.canonical_property_id = canonical
    db.upsert_listing(best)
    for other, _ in hits[1:]:
        other.canonical_property_id = canonical
        db.upsert_listing(other)

    record.duplicate_of = canonical
    db.save_call(record)
    return {"duplicate_of": canonical, "duplicate_reasons": result.reasons}


def _persist(state: CallState) -> dict:
    record = state["record"]
    assert record is not None
    listing = db.get_listing(record.listing_id)
    if listing and record.verification_status:
        listing.verification_status = record.verification_status
        db.upsert_listing(listing)
    return {}


def _summarize(state: CallState) -> dict:
    record = state["record"]
    assert record is not None
    f = record.facts
    parts: list[str] = []
    if f.is_available is True:
        parts.append("Broker confirmed the flat is available")
    elif f.is_available is False:
        parts.append("Broker said the flat is no longer available")
    else:
        parts.append("Broker did not clearly confirm availability")
    if f.rent:
        parts.append(f"rent Rs {f.rent:,}/month")
    if f.deposit:
        parts.append(f"deposit Rs {f.deposit:,}")
    if f.available_from:
        parts.append(f"available from {f.available_from.isoformat()}")
    if f.visit_slots:
        parts.append("visit slots: " + ", ".join(f.visit_slots))
    if state.get("duplicate_of"):
        parts.append(
            f"WARNING: same property also listed under {state['duplicate_of']} "
            f"({'; '.join(state['duplicate_reasons'])})"
        )
    record.summary = "; ".join(parts) + "."
    db.save_call(record)
    return {"record": record}


def build_graph():
    g = StateGraph(CallState)
    g.add_node("normalize_facts", _normalize_facts)
    g.add_node("verify_listing", _verify_listing)
    g.add_node("resolve_entities", _resolve_entities)
    g.add_node("persist", _persist)
    g.add_node("summarize", _summarize)
    g.add_edge(START, "normalize_facts")
    g.add_edge("normalize_facts", "verify_listing")
    g.add_edge("verify_listing", "resolve_entities")
    g.add_edge("resolve_entities", "persist")
    g.add_edge("persist", "summarize")
    g.add_edge("summarize", END)
    return g.compile()


def run_post_call(call_id: str) -> CallState:
    graph = build_graph()
    record = db.get_call(call_id)
    if record is None:
        raise ValueError(f"unknown call {call_id}")
    return graph.invoke(
        {"call_id": call_id, "record": record, "duplicate_of": None, "duplicate_reasons": []}
    )
