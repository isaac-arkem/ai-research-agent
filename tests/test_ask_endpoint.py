from unittest.mock import patch
from uuid import uuid4

from app.core.config import get_settings
from app.main import app
from app.models.domain import AgentResult, ValidationError_, ValidationResult
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


@patch("app.api.v1.endpoints.research.generate_research_plan")
def test_conversation_round_trip(mock_gen, client):
    mock_gen.return_value = _ok()
    created = client.post("/ask", json={"prompt": "Find modest fashion creators in SA"})
    cid = created.json()["conversation_id"]
    loaded = client.get(f"/conversations/{cid}")
    assert loaded.status_code == 200
    messages = loaded.json()["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant"]


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["markets"] >= 1
    assert "auth_required" in body


def test_ask_openapi_success_schema_does_not_repeat_plan(client):
    spec = client.get("/openapi.json").json()
    schema = spec["components"]["schemas"]["AskResponse"]
    assert "plan" in schema["properties"]
    assert "validation" not in schema["properties"]
    assert "raw" not in schema["properties"]
    assert "error" not in schema["properties"]


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
    assert third.status_code == 429
    assert third.json()["code"] == "rate_limited"
    assert "Retry-After" in third.headers
