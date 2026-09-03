import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-real")
os.environ["AUTH_REQUIRED"] = "false"

import pytest
from fastapi.testclient import TestClient

from app.core.rate_limit import limiter
from app.main import app
from app.services.conversations import MemoryConversationStore
from app.services import conversations as conversation_service


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    limiter.reset()
    mem = MemoryConversationStore()
    monkeypatch.setattr(conversation_service, "store", mem)
    monkeypatch.setattr("app.services.audit.get_supabase_admin", lambda *a, **k: None)
    yield
    limiter.reset()
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client
