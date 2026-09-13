"""SQLite store + mock seed data.

PROTOTYPE DATA: every seeded listing is invented. Three of them are the same
flat listed by three different brokers (the classic Indian rental-market
duplicate problem) so the entity-resolution path has something real to catch.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .schemas import CallFacts, CallRecord, Listing, TurnLatency

DB_PATH = Path(__file__).resolve().parent / "broker_agent.db"


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(path: Path | None = None) -> None:
    global DB_PATH
    if path is not None:
        DB_PATH = path
    conn = _conn()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS listings (
            id TEXT PRIMARY KEY,
            data TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS calls (
            id TEXT PRIMARY KEY,
            listing_id TEXT NOT NULL,
            data TEXT NOT NULL
        );
        """
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------- listings

def upsert_listing(listing: Listing) -> None:
    conn = _conn()
    conn.execute(
        "INSERT INTO listings (id, data) VALUES (?, ?) "
        "ON CONFLICT(id) DO UPDATE SET data=excluded.data",
        (listing.id, listing.model_dump_json()),
    )
    conn.commit()
    conn.close()


def get_listing(listing_id: str) -> Listing | None:
    conn = _conn()
    row = conn.execute("SELECT data FROM listings WHERE id=?", (listing_id,)).fetchone()
    conn.close()
    return Listing.model_validate_json(row["data"]) if row else None


def list_listings() -> list[Listing]:
    conn = _conn()
    rows = conn.execute("SELECT data FROM listings").fetchall()
    conn.close()
    return [Listing.model_validate_json(r["data"]) for r in rows]


# ---------------------------------------------------------------- calls

def new_call(listing_id: str) -> CallRecord:
    record = CallRecord(
        id=str(uuid.uuid4())[:8],
        listing_id=listing_id,
        started_at=datetime.now(timezone.utc),
    )
    save_call(record)
    return record


def save_call(record: CallRecord) -> None:
    conn = _conn()
    conn.execute(
        "INSERT INTO calls (id, listing_id, data) VALUES (?, ?, ?) "
        "ON CONFLICT(id) DO UPDATE SET data=excluded.data",
        (record.id, record.listing_id, record.model_dump_json()),
    )
    conn.commit()
    conn.close()


def get_call(call_id: str) -> CallRecord | None:
    conn = _conn()
    row = conn.execute("SELECT data FROM calls WHERE id=?", (call_id,)).fetchone()
    conn.close()
    return CallRecord.model_validate_json(row["data"]) if row else None


def list_calls() -> list[CallRecord]:
    conn = _conn()
    rows = conn.execute("SELECT data FROM calls").fetchall()
    conn.close()
    return [CallRecord.model_validate_json(r["data"]) for r in rows]


# ---------------------------------------------------------------- seed data

def seed_mock_listings() -> int:
    """Insert the invented demo listings. Returns how many were seeded."""
    from datetime import date

    seeds = [
        # --- the same Sobha City flat, listed by three brokers ---------------
        dict(id="L-1001", source="99acres", broker_name="Rakesh", broker_phone="+91 98100 11111",
             society="Sobha City", address="Tower 4, Flat 1203, Sobha City, Sector 108",
             sector="Sector 108", city="Gurgaon", bhk=2, rent=34000, deposit=68000,
             available_from=date(2026, 10, 1)),
        dict(id="L-1002", source="magicbricks", broker_name="Imran", broker_phone="+91 98100 22222",
             society="Sobha City", address="Sobha City T-4 1203, Sec 108",
             sector="108", city="Gurugram", bhk=2, rent=35500, deposit=None,
             available_from=date(2026, 10, 1)),
        dict(id="L-1003", source="broker-whatsapp", broker_name="Rakesh", broker_phone="+91 98100 11111",
             society="Sobha City", address="12th floor 2BHK Sobha city sec-108",
             sector="Sec 108", city="Gurgaon", bhk=2, rent=35000, deposit=70000,
             available_from=None),
        # --- genuinely different flats ----------------------------------------
        dict(id="L-1004", source="nobroker", broker_name="Owner: Meena", broker_phone="+91 98100 33333",
             society="DLF Phase 3 independent floor", address="Block C, DLF Phase 3",
             sector="DLF Phase 3", city="Gurgaon", bhk=1, rent=22000, deposit=44000,
             available_from=date(2026, 9, 20)),
        dict(id="L-1005", source="99acres", broker_name="Sana", broker_phone="+91 98100 44444",
             society="Prestige Falcon City", address="Tower 9, Flat 404, Konanakunte",
             sector="Kanakapura Road", city="Bengaluru", bhk=2, rent=31000, deposit=93000,
             available_from=date(2026, 10, 5)),
        dict(id="L-1006", source="magicbricks", broker_name="Vikram", broker_phone="+91 98100 55555",
             society="HSR Layout independent house", address="27th Main, HSR Sector 2",
             sector="HSR", city="Bengaluru", bhk=1, rent=19500, deposit=58500,
             available_from=date(2026, 9, 25)),
        dict(id="L-1007", source="broker-whatsapp", broker_name="Imran", broker_phone="+91 98100 22222",
             society="Godrej Air", address="Godrej Air, Sector 85",
             sector="Sector 85", city="Gurgaon", bhk=3, rent=48000, deposit=96000,
             available_from=date(2026, 10, 15)),
        dict(id="L-1008", source="99acres", broker_name="Pooja", broker_phone="+91 98100 66666",
             society="Sobha Dream Acres", address="Wing B, Flat 812, Panathur",
             sector="Panathur", city="Bengaluru", bhk=2, rent=33000, deposit=99000,
             available_from=date(2026, 10, 1)),
    ]
    now = datetime.now(timezone.utc)
    count = 0
    for s in seeds:
        if get_listing(s["id"]) is None:
            upsert_listing(Listing(listed_at=now, **s))
            count += 1
    return count


# re-export for convenience in tests
__all__ = [
    "init_db", "upsert_listing", "get_listing", "list_listings",
    "new_call", "save_call", "get_call", "list_calls", "seed_mock_listings",
    "CallFacts", "CallRecord", "TurnLatency", "json",
]
