from unittest.mock import patch
from uuid import uuid4

from app.core.config import get_settings
from app.main import app
from app.models.domain import AgentResult, ValidationError_, ValidationResult
from app.models.requests import MAX_HISTORY_CONTENT, MAX_HISTORY_TURNS
from tests.plans import discovery_plan, success_result


def _ok():
    plan = discovery_plan()
    return success_result(plan, "discovery")


@patch("app.api.v1.endpoints.research.generate_research_plan")
def test_ask_success_input_output(mock_gen, client):
    mock_gen.return_value = _ok()
    response = client.post("/ask", json={"prompt": "Find modest fashion creators in Saudi Arabia"})
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["flow"] == "discovery"
    assert body["conversation_id"]
    assert body["plan"]["recommended_runs"][0]["countries"] == ["SA"]
    assert "X-RateLimit-Limit" in response.headers


@patch("app.api.v1.endpoints.research.generate_research_plan")
def test_follow_up_passes_conversation_history(mock_gen, client):
    mock_gen.return_value = _ok()
    first = client.post("/ask", json={"prompt": "Find fashion creators in SA"})
    cid = first.json()["conversation_id"]
    mock_gen.return_value = _ok()
    second = client.post(
        "/ask",
        json={"prompt": "Make it Instagram only", "conversation_id": cid},
    )
    assert second.status_code == 200
    history = mock_gen.call_args.kwargs["history"]
    assert len(history) >= 2
    assert history[0].role == "user"
    assert history[1].role == "assistant"


@patch("app.api.v1.endpoints.research.generate_research_plan")
def test_client_conversation_history_is_passed(mock_gen, client):
    mock_gen.return_value = _ok()
    response = client.post(
        "/ask",
        json={
            "prompt": "Make it Instagram only",
            "conversation_history": [
                {"role": "user", "content": "Find fashion in SA"},
                {"role": "assistant", "content": "Plan for SA modest fashion."},
            ],
        },
    )
    assert response.status_code == 200
    history = mock_gen.call_args.kwargs["history"]
    assert [t.content for t in history] == [
        "Find fashion in SA",
        "Plan for SA modest fashion.",
    ]


@patch("app.api.v1.endpoints.research.generate_research_plan")
def test_swagger_empty_conversation_id_is_treated_as_new_thread(mock_gen, client):
    mock_gen.return_value = _ok()
    response = client.post(
        "/ask",
        json={
            "prompt": "influencers in nigeria",
            "model": "gpt-4o",
            "conversation_id": "",
            "conversation_history": [{"role": "user", "content": "string"}],
        },
    )
    assert response.status_code == 200
    assert response.json()["ok"] is True
    mock_gen.assert_called_once()
    assert mock_gen.call_args.kwargs["history"] == []


@patch("app.api.v1.endpoints.research.generate_research_plan")
def test_unknown_conversation_is_400(mock_gen, client):
    response = client.post(
        "/ask",
        json={"prompt": "hello", "conversation_id": str(uuid4())},
    )
    assert response.status_code == 400
    assert response.json()["code"] == "unknown_conversation"
    mock_gen.assert_not_called()


@patch("app.api.v1.endpoints.research.generate_research_plan")
def test_empty_sanitized_prompt_is_400(mock_gen, client):
    mock_gen.return_value = AgentResult(
        ok=False, error="Empty prompt after sanitisation", error_code="empty_prompt"
    )
    response = client.post("/ask", json={"prompt": "   "})
    assert response.status_code == 400
    assert response.json()["code"] == "empty_prompt"


def test_missing_prompt_is_422(client):
    response = client.post("/ask", json={})
    assert response.status_code == 422
    body = response.json()
    assert body["ok"] is False
    assert body["code"] == "validation_error"


@patch("app.api.v1.endpoints.research.generate_research_plan")
def test_invalid_plan_is_422(mock_gen, client):
    mock_gen.return_value = AgentResult(
        ok=False,
        error="Validation failed: niche must be a slug",
        error_code="validation_failed",
        validation=ValidationResult(
            valid=False,
            errors=[ValidationError_(rule=6, field="niche", message="bad niche")],
        ),
    )
    response = client.post("/ask", json={"prompt": "find creators"})
    assert response.status_code == 422
    assert response.json()["code"] == "validation_failed"
    assert response.json()["details"][0]["rule"] == 6


@patch("app.api.v1.endpoints.research.generate_research_plan")
def test_openai_failure_is_502(mock_gen, client):
    mock_gen.return_value = AgentResult(
        ok=False, error="OpenAI call failed: timeout", error_code="openai_failed"
    )
    response = client.post("/ask", json={"prompt": "find creators"})
    assert response.status_code == 502
    assert response.json()["code"] == "openai_failed"
    assert response.json()["error"] == "Upstream failed"
    assert "timeout" not in response.json()["error"]


@patch("app.api.v1.endpoints.research.generate_research_plan")
def test_client_model_is_ignored(mock_gen, client):
    mock_gen.return_value = _ok()
    response = client.post(
        "/ask", json={"prompt": "find creators", "model": "o1"}
    )
    assert response.status_code == 200
    assert mock_gen.call_args.kwargs["model"] == get_settings().research_agent_model


def test_history_turn_over_cap_is_422(client):
    response = client.post(
        "/ask",
        json={
            "prompt": "find creators",
            "conversation_history": [
                {"role": "user", "content": "x" * (MAX_HISTORY_CONTENT + 1)},
            ],
        },
    )
    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"


@patch("app.api.v1.endpoints.research.generate_research_plan")
def test_a_plan_sized_history_turn_is_kept(mock_gen, client):
    """A previous plan is thousands of characters. It must survive intact."""
    mock_gen.return_value = _ok()
    prior = "Plan for SA modest fashion. " * 100  # ~2.8k
    assert len(prior) > 2000
    response = client.post(
        "/ask",
        json={
            "prompt": "Make it Instagram only",
            "conversation_history": [
                {"role": "user", "content": "Find fashion in SA"},
                {"role": "assistant", "content": prior},
            ],
        },
    )
    assert response.status_code == 200
    assert mock_gen.call_args.kwargs["history"][1].content == prior


@patch("app.api.v1.endpoints.research.generate_research_plan")
def test_a_long_brief_prompt_is_accepted(mock_gen, client):
    """Operators paste a situation before the ask. 2000 chars was too tight."""
    mock_gen.return_value = _ok()
    brief = (
        "We already scrape modest fashion in SA on Instagram. "
        "Now we want TikTok in the same niche, same lookback, "
        "but exclude anything that looks like a storefront. "
    ) * 80
    assert 2000 < len(brief) < 32_000
    response = client.post("/ask", json={"prompt": brief})
    assert response.status_code == 200
    assert mock_gen.call_args.args[0] == brief


def test_history_list_over_cap_is_422(client):
    response = client.post(
        "/ask",
        json={
            "prompt": "find creators",
            "conversation_history": [
                {"role": "user", "content": f"turn {i}"}
                for i in range(MAX_HISTORY_TURNS + 1)
            ],
        },
    )
    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"


@patch("app.api.v1.endpoints.research.generate_research_plan")
def test_conversation_round_trip(mock_gen, client):
    mock_gen.return_value = _ok()
    created = client.post("/ask", json={"prompt": "Find modest fashion creators in SA"})
    cid = created.json()["conversation_id"]
    loaded = client.get(f"/conversations/{cid}")
    assert loaded.status_code == 200
    messages = loaded.json()["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant"]


def test_list_conversations_empty(client):
    listed = client.get("/conversations")
    assert listed.status_code == 200
    assert listed.json()["conversations"] == []


@patch("app.api.v1.endpoints.research.generate_research_plan")
def test_list_conversations_newest_first(mock_gen, client):
    mock_gen.return_value = _ok()
    first = client.post("/ask", json={"prompt": "Find modest fashion creators in SA"})
    cid1 = first.json()["conversation_id"]
    second = client.post("/ask", json={"prompt": "Scrape @khloekardashian on Instagram"})
    cid2 = second.json()["conversation_id"]
    listed = client.get("/conversations")
    assert listed.status_code == 200
    items = listed.json()["conversations"]
    ids = [c["id"] for c in items]
    assert ids[0] == cid2
    assert cid1 in ids
    titles = {c["id"]: c.get("title") for c in items}
    assert "modest fashion" in (titles[cid1] or "").lower()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["markets"] >= 1
    assert "auth_required" in body


def test_docs_are_off_outside_debug(client):
    assert client.get("/docs").status_code == 404
    assert client.get("/redoc").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_ask_openapi_success_schema_does_not_repeat_plan():
    from fastapi.testclient import TestClient

    from app.main import create_app

    debug_app = create_app(get_settings().model_copy(update={"debug": True}))
    with TestClient(debug_app) as debug_client:
        spec = debug_client.get("/openapi.json").json()
    schema = spec["components"]["schemas"]["AskResponse"]
    assert "plan" in schema["properties"]
    assert "validation" not in schema["properties"]
    assert "raw" not in schema["properties"]
    assert "error" not in schema["properties"]
    assert "model" not in spec["components"]["schemas"]["AskRequest"]["properties"]


def test_chat_ui(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Research Agent" in response.text


@patch("app.api.v1.endpoints.research.generate_research_plan")
def test_rate_limit_429(mock_gen, client):
    mock_gen.return_value = _ok()
    settings = get_settings().model_copy(update={"rate_limit_requests": 2})
    app.dependency_overrides[get_settings] = lambda: settings
    assert client.post("/ask", json={"prompt": "one"}).status_code == 200
    assert client.post("/ask", json={"prompt": "two"}).status_code == 200
    third = client.post("/ask", json={"prompt": "three"})
    assert "Retry-After" in third.headers
