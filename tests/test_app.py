import os
import sys
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app import db  # noqa: E402


@pytest.fixture()
def client():
    with tempfile.TemporaryDirectory() as tmp:
        db.init_db(Path(tmp) / "t.db")
        from app.main import app
        with TestClient(app) as c:
            yield c


def test_listings_seeded(client):
    items = client.get("/api/listings").json()
    assert len(items) >= 8
    assert any(l["society"] == "Sobha City" for l in items)


def test_offer_503_without_keys(client):
    call = client.post("/api/calls", params={"listing_id": "L-1001"}).json()
    os.environ.pop("SARVAM_API_KEY", None)
    os.environ.pop("GROQ_API_KEY", None)
    res = client.post("/api/offer", json={"sdp": "x", "type": "offer", "call_id": call["id"]})
    assert res.status_code == 503
    assert "SARVAM_API_KEY" in res.json()["detail"]


def test_full_call_lifecycle_without_voice(client):
    call = client.post("/api/calls", params={"listing_id": "L-1001"}).json()
    rec = db.get_call(call["id"])
    rec.facts.is_available = True
    rec.facts.rent = 34000
    db.save_call(rec)
    out = client.post(f"/api/calls/{call['id']}/process").json()
    assert out["verification_status"] == "verified_available"
    assert out["duplicate_of"] is not None
    assert "available" in out["summary"]
    # listings endpoint now reflects the verification
    after = {l["id"]: l for l in client.get("/api/listings").json()}
    assert after["L-1001"]["verification_status"] == "verified_available"


def test_index_served(client):
    res = client.get("/")
    assert res.status_code == 200 and b"Broker-Voice-Agent" in res.content
