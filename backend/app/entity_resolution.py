"""Entity resolution: decide whether two listings describe the same flat.

Indian rental portals are full of the same property listed by multiple
brokers with slightly different spellings, rents, and sector formats
("Sector 108" vs "Sec 108" vs "108"). This module scores candidate pairs on
weighted evidence and explains its reasoning - no black box.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from .schemas import Listing


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(sector|sec|phase|tower|t|flat|floor|blk|block)\b", " ", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _sector_number(sector: str) -> str | None:
    m = re.search(r"\d+", sector)
    return m.group(0) if m else None


_CITY_ALIASES = {"gurgaon": "gurugram", "bangalore": "bengaluru"}


def _canon_city(city: str) -> str:
    c = city.strip().lower()
    return _CITY_ALIASES.get(c, c)


@dataclass
class MatchResult:
    score: float  # 0..1
    reasons: list[str] = field(default_factory=list)

    @property
    def is_duplicate(self) -> bool:
        return self.score >= 0.75


def score_pair(a: Listing, b: Listing) -> MatchResult:
    """Weighted evidence score that listings a and b are the same property."""
    reasons: list[str] = []
    score = 0.0

    if _canon_city(a.city) != _canon_city(b.city):
        return MatchResult(0.0, ["different cities"])

    society_sim = fuzz.token_set_ratio(_normalize(a.society), _normalize(b.society)) / 100
    if society_sim >= 0.85:
        score += 0.40
        reasons.append(f"society match ({a.society!r} ~ {b.society!r})")
    elif society_sim >= 0.60:
        score += 0.20
        reasons.append("partial society match")

    addr_sim = fuzz.token_set_ratio(_normalize(a.address), _normalize(b.address)) / 100
    if addr_sim >= 0.70:
        score += 0.25
        reasons.append("address overlap")
    elif addr_sim >= 0.45:
        score += 0.10

    sa, sb = _sector_number(a.sector), _sector_number(b.sector)
    if sa and sb and sa == sb:
        score += 0.15
        reasons.append(f"same sector ({sa})")
    elif _normalize(a.sector) and _normalize(a.sector) == _normalize(b.sector):
        score += 0.15
        reasons.append("same sector name")

    if a.bhk == b.bhk:
        score += 0.10
        reasons.append(f"both {a.bhk}BHK")

    if a.rent and b.rent:
        lo, hi = min(a.rent, b.rent), max(a.rent, b.rent)
        if hi - lo <= 0.10 * hi:
            score += 0.10
            reasons.append(f"rent within 10% ({a.rent} vs {b.rent})")

    if a.broker_phone == b.broker_phone:
        score += 0.05
        reasons.append("same broker phone")

    return MatchResult(min(score, 1.0), reasons)


def find_duplicates(target: Listing, others: list[Listing]) -> list[tuple[Listing, MatchResult]]:
    """All listings in `others` that look like the same property as `target`."""
    hits = []
    for other in others:
        if other.id == target.id:
            continue
        result = score_pair(target, other)
        if result.is_duplicate:
            hits.append((other, result))
    hits.sort(key=lambda h: h[1].score, reverse=True)
    return hits
