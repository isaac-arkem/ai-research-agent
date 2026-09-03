from unittest.mock import Mock, patch

from app.core.config import get_settings
from app.main import app


def test_auth_required_without_bearer_is_401(client):
    settings = get_settings().model_copy(
        update={
            "auth_required": True,
            "next_public_supabase_url": "https://example.supabase.co",
            "next_public_supabase_anon_key": "anon",
        }
    )
    app.dependency_overrides[get_settings] = lambda: settings
    response = client.post("/ask", json={"prompt": "find creators"})
    assert response.status_code == 401
    assert response.json()["code"] == "auth_required"


@patch("app.core.auth.httpx.get")
@patch("app.api.v1.endpoints.research.generate_research_plan")
def test_auth_required_with_valid_bearer(mock_gen, mock_get, client):
    from tests.plans import discovery_plan, success_result

    plan = discovery_plan()
    mock_gen.return_value = success_result(plan, "discovery")
    mock_get.return_value = Mock(status_code=200, json=lambda: {"id": "user-123", "email": "a@b.c"})

    settings = get_settings().model_copy(
        update={
            "auth_required": True,
            "next_public_supabase_url": "https://example.supabase.co",
            "next_public_supabase_anon_key": "anon",
        }
    )
    app.dependency_overrides[get_settings] = lambda: settings

    response = client.post(
        "/ask",
        json={"prompt": "Find modest fashion creators in SA"},
        headers={"Authorization": "Bearer fake-jwt"},
    )
    assert response.status_code == 200
    mock_get.assert_called()
    assert mock_get.call_args.kwargs["headers"]["Authorization"] == "Bearer fake-jwt"


@patch("app.core.auth.httpx.get")
def test_auth_required_invalid_token_is_401(mock_get, client):
    mock_get.return_value = Mock(status_code=401, json=lambda: {})
    settings = get_settings().model_copy(
        update={
            "auth_required": True,
            "next_public_supabase_url": "https://example.supabase.co",
            "next_public_supabase_anon_key": "anon",
        }
    )
    app.dependency_overrides[get_settings] = lambda: settings
    response = client.post(
        "/ask",
        json={"prompt": "find creators"},
        headers={"Authorization": "Bearer bad"},
    )
    assert response.status_code == 401
    assert response.json()["code"] == "invalid_token"
