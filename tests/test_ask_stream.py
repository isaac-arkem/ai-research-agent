from unittest.mock import patch

from app.models.domain import AgentResult
from tests.plans import discovery_plan, success_result


def _parse_sse(lines):
    events = []
    for line in lines:
        if line.startswith("event: "):
            events.append(line[len("event: "):])
    return events


@patch("app.api.v1.endpoints.research.generate_research_plan")
def test_ask_stream_success_narrates_progress_then_done(mock_gen, client):
    def fake_gen(prompt, ctx, **kwargs):
        on_progress = kwargs.get("on_progress")
        if on_progress:
            on_progress("triaged", action="search")
            on_progress("searching", query=prompt, country="SA", detail="Searching web")
        return success_result(discovery_plan(), "discovery")

    mock_gen.side_effect = fake_gen

    with client.stream(
        "POST", "/ask/stream",
        json={"prompt": "Find modest fashion creators in Saudi Arabia"},
    ) as response:
        assert response.status_code == 200
        lines = list(response.iter_lines())

    assert _parse_sse(lines) == ["progress", "progress", "progress", "done"]


@patch("app.api.v1.endpoints.research.generate_research_plan")
def test_ask_stream_failure_narrates_progress_then_failed(mock_gen, client):
    def fake_gen(prompt, ctx, **kwargs):
        on_progress = kwargs.get("on_progress")
        if on_progress:
            on_progress("triaged", action="search")
        return AgentResult(ok=False, error="boom", error_code="openai_failed")

    mock_gen.side_effect = fake_gen

    with client.stream(
        "POST", "/ask/stream",
        json={"prompt": "Find modest fashion creators in Saudi Arabia"},
    ) as response:
        assert response.status_code == 200
        lines = list(response.iter_lines())

    assert _parse_sse(lines) == ["progress", "progress", "failed"]


@patch("app.api.v1.endpoints.research.generate_research_plan")
def test_ask_stream_crash_hides_exception(mock_gen, client):
    mock_gen.side_effect = RuntimeError("secret internals")

    with client.stream(
        "POST", "/ask/stream",
        json={"prompt": "Find modest fashion creators in Saudi Arabia"},
    ) as response:
        assert response.status_code == 200
        blob = "\n".join(response.iter_lines())

    assert "secret internals" not in blob
    assert "Upstream failed" in blob
