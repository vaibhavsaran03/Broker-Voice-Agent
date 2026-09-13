"""Tests for everything that does not need live API keys.

The voice pipeline itself (Sarvam/Groq) is exercised manually with keys set;
the verification logic around it is deterministic and fully covered here.
"""
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app import db, graph  # noqa: E402
from app.entity_resolution import find_duplicates, score_pair  # noqa: E402
from app.pipeline import _parse_fact  # noqa: E402
from app.schemas import CallFacts, VerificationStatus  # noqa: E402


@pytest.fixture()
def fresh_db():
    with tempfile.TemporaryDirectory() as tmp:
        db.init_db(Path(tmp) / "test.db")
        db.seed_mock_listings()
        yield


# ------------------------------------------------------------- entity resolution

def test_sobha_triplicates_match_each_other(fresh_db):
    listings = {l.id: l for l in db.list_listings()}
    for a, b in [("L-1001", "L-1002"), ("L-1001", "L-1003"), ("L-1002", "L-1003")]:
        result = score_pair(listings[a], listings[b])
        assert result.is_duplicate, f"{a} vs {b}: {result.score} {result.reasons}"


def test_different_flats_do_not_match(fresh_db):
    listings = {l.id: l for l in db.list_listings()}
    assert not score_pair(listings["L-1001"], listings["L-1004"]).is_duplicate
    assert not score_pair(listings["L-1001"], listings["L-1005"]).is_duplicate  # different city
    assert not score_pair(listings["L-1005"], listings["L-1008"]).is_duplicate  # both Blr 2BHK


def test_find_duplicates_returns_the_two_other_copies(fresh_db):
    listings = {l.id: l for l in db.list_listings()}
    hits = find_duplicates(listings["L-1001"], list(listings.values()))
    assert {h[0].id for h in hits} == {"L-1002", "L-1003"}


# ------------------------------------------------------------------ fact parsing

def _record():
    return db.CallRecord(id="t1", listing_id="L-1001", started_at=datetime.now(timezone.utc))


def test_parse_fact_numbers(fresh_db):
    r = _record()
    assert _parse_fact(r, "rent", "35k") == "recorded" and r.facts.rent == 35000
    assert _parse_fact(r, "rent", "Rs 34,000") == "recorded" and r.facts.rent == 34000
    # month-based deposits convert against recorded rent (or fall back to a note)
    assert r.facts.rent == 34000
    assert _parse_fact(r, "deposit", "2 months") == "recorded" and r.facts.deposit == 68000
    r2 = _record()  # no rent recorded yet
    assert _parse_fact(r2, "deposit", "2 months") == "recorded" and r2.facts.deposit is None
    assert r2.facts.notes == ["deposit: 2 months rent"]
    assert _parse_fact(r, "available_from", "2026-10-01") == "recorded"
    assert r.facts.available_from == date(2026, 10, 1)
    assert _parse_fact(r, "rent", "around thirty five") != "recorded"


def test_parse_fact_availability(fresh_db):
    r = _record()
    _parse_fact(r, "is_available", "yes")
    assert r.facts.is_available is True
    _parse_fact(r, "is_available", "not available")
    assert r.facts.is_available is False


# ------------------------------------------------------------------ post-call graph

def _run_graph(fresh_db, facts: CallFacts, listing_id="L-1001"):
    record = db.new_call(listing_id)
    record.facts = facts
    record.transcript = [{"role": "broker", "text": "haan available hai", "ts": "t"}]
    db.save_call(record)
    return graph.run_post_call(record.id)


def test_graph_verifies_available_and_flags_duplicate(fresh_db):
    state = _run_graph(fresh_db, CallFacts(is_available=True, rent=34000, deposit=68000,
                                           available_from=date(2026, 10, 1),
                                           visit_slots=["Sat 11am"]))
    record = state["record"]
    assert record.verification_status == VerificationStatus.VERIFIED_AVAILABLE
    assert state["duplicate_of"] is not None
    assert "society match" in " ".join(state["duplicate_reasons"])
    assert "available" in record.summary
    # listing itself got updated
    assert db.get_listing("L-1001").verification_status == VerificationStatus.VERIFIED_AVAILABLE
    # all three copies share one canonical property id
    props = {db.get_listing(i).canonical_property_id for i in ("L-1001", "L-1002", "L-1003")}
    assert len(props) == 1 and props.pop() == state["duplicate_of"]


def test_graph_marks_rent_conflict(fresh_db):
    state = _run_graph(fresh_db, CallFacts(is_available=True, rent=45000))
    assert state["record"].verification_status == VerificationStatus.CONFLICT


def test_graph_marks_unavailable_and_stale(fresh_db):
    assert _run_graph(fresh_db, CallFacts(is_available=False))["record"].verification_status \
        == VerificationStatus.VERIFIED_UNAVAILABLE
    assert _run_graph(fresh_db, CallFacts())["record"].verification_status == VerificationStatus.STALE


def test_deposit_derived_only_from_explicit_note(fresh_db):
    facts = CallFacts(is_available=True, rent=30000, notes=["deposit 2 months deposit"])
    state = _run_graph(fresh_db, facts)
    assert state["record"].facts.deposit == 60000
