import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-real")
os.environ["AUTH_REQUIRED"] = "false"

import pytest
from fastapi.testclient import TestClient

from app.core.dependencies import reset_agent_context
from app.core.rate_limit import limiter
from app.main import app
from app.models.domain import MarketEntry, TaxonomyEntry
from app.services.conversations import MemoryConversationStore
from app.services import conversations as conversation_service

# The country list now comes from apify_supported_countries with no hardcoded
# fallback, so tests must supply their own. Previously they leaned on
# app/data/markets.py by accident, which meant the suite quietly depended on
# production reference data. This keeps them hermetic and offline.
TEST_COUNTRIES = [
    MarketEntry(code=c, iso=c, name=n)
    for c, n in [
        ("SA", "Saudi Arabia"), ("AE", "United Arab Emirates"),
        ("KW", "Kuwait"), ("EG", "Egypt"), ("MA", "Morocco"),
        ("NG", "Nigeria"), ("ZA", "South Africa"), ("BR", "Brazil"),
        ("MX", "Mexico"), ("CO", "Colombia"), ("AR", "Argentina"),
        ("ID", "Indonesia"), ("PH", "Philippines"), ("TH", "Thailand"),
        ("MY", "Malaysia"), ("JP", "Japan"), ("TR", "Turkey"),
        ("AM", "Armenia"), ("IN", "India"),
    ]
]

TEST_NICHES = [
    TaxonomyEntry(slug=s, aliases=a)
    for s, a in [
        ("music_dance", []), ("cooking_mum", ["Cooking Mum"]),
        ("comedy_skits", []), ("fashion_beauty", []), ("trading", []),
        ("spirituality", []), ("travel", []), ("beauty", []),
    ]
]


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    limiter.reset()
    mem = MemoryConversationStore()
    monkeypatch.setattr(conversation_service, "store", mem)
    monkeypatch.setattr("app.services.audit.get_supabase_admin", lambda *a, **k: None)
    monkeypatch.setattr("app.services.known_accounts.get_supabase_admin", lambda *a, **k: None)
    monkeypatch.setattr(
        "app.core.dependencies._fetch_countries_from_db",
        lambda *a, **k: list(TEST_COUNTRIES),
    )
    # Niches come from the DB too now. Without this the suite silently reaches
    # production Supabase — which is slow, and couples test outcomes to
    # whatever niches happen to exist that day.
    monkeypatch.setattr(
        "app.core.dependencies._fetch_taxonomy_from_db",
        lambda *a, **k: list(TEST_NICHES),
    )
    reset_agent_context()
    yield
    reset_agent_context()
    limiter.reset()
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client
