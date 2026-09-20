"""The tool boundary — what the model may do, and what it may spend.

The model decides WHAT to do. These decide what it is allowed to COST, and
that split is the whole reason the guardrail lives here rather than in the
flow: once a model can call a scrape, a rule in a prompt is advice.
"""

from types import SimpleNamespace
from unittest.mock import patch

from app.models.domain import AgentContext, ChatTurn, MarketEntry, TaxonomyEntry
from app.services.tools import TERMINAL, TOOL_SCHEMAS, ToolContext, run_tool


def _ctx():
    return AgentContext(
        markets=[MarketEntry(code="GH", iso="GH", name="Ghana")],
        taxonomy=[TaxonomyEntry(slug="music", aliases=["music"])],
    )


def _settings(**kw):
    base = dict(openai_api_key="sk-test", grounding_model="gpt-4o-mini",
                research_plan_model="gpt-4o", search_timeout=15.0,
                search_results_per_query=5, research_window_days=365,
                research_depth="default", tools_max_paid_calls=2,
                tools_max_searches=4)
    base.update(kw)
    return SimpleNamespace(**base)


def _tc(prompt, history=None, **kw):
    return ToolContext(settings=_settings(**kw), ctx=_ctx(), prompt=prompt, history=history)


def test_a_scrape_is_refused_unless_the_operator_named_the_platform():
    """Not "discouraged in the prompt" — refused, with Apify never reached."""
    tc = _tc("find dancehall creators in Ghana")
    with patch("app.services.research.orchestrator.run_research") as apify:
        out = run_tool("search_social", {"platform": "instagram", "query": "x"}, tc)

    assert not apify.called
    assert tc.paid_calls == 0
    assert "refused" in out and "not asked for instagram" in out
    # The refusal has to tell the model what to do instead, or it just retries.
    assert "Ask them" in out


def test_the_platform_may_be_named_anywhere_in_the_thread():
    history = [ChatTurn(role="user", content="find dancehall creators on instagram")]
    tc = _tc("the ones from Accra", history=history)
    with patch("app.services.research.orchestrator.run_research") as apify:
        with patch("app.services.research.orchestrator.resolve_targets", return_value={}):
            run_tool("search_social", {"platform": "instagram", "query": "x"}, tc)
    assert apify.called


def test_the_paid_budget_is_a_wall():
    tc = _tc("creators on instagram", tools_max_paid_calls=1)
    with patch("app.services.research.orchestrator.run_research") as apify:
        with patch("app.services.research.orchestrator.resolve_targets", return_value={}):
            run_tool("search_social", {"platform": "instagram", "query": "a"}, tc)
            first = apify.call_count
            out = run_tool("search_social", {"platform": "instagram", "query": "b"}, tc)

    assert first == 1
    assert apify.call_count == 1, "the second call must not reach Apify"
    assert "budget for this turn is spent" in out


def test_an_unknown_platform_is_not_a_crash():
    tc = _tc("creators on myspace")
    assert "no lane for" in run_tool("search_social", {"platform": "myspace", "query": "x"}, tc)


def test_a_failing_tool_returns_a_sentence_not_an_exception():
    """A broken tool must not break the turn — the model reads the failure."""
    tc = _tc("anything")
    assert "no tool called" in run_tool("nonexistent", {}, tc)
    assert "wrong arguments" in run_tool("search_web", {"nope": 1}, tc)
    with patch("app.services.grounding.resolve_seed", side_effect=RuntimeError("boom")):
        assert "failed" in run_tool("resolve_account", {"name": "Sarkodie"}, tc)


def test_searching_is_capped_too():
    tc = _tc("anything", tools_max_searches=1)
    tc.searches = 1
    assert "already searched" in run_tool("search_web", {"query": "x"}, tc)


def test_every_tool_is_dispatchable_or_terminal():
    """A schema the model can see but nothing can run is a dead end."""
    from app.services.tools import _DISPATCH

    for schema in TOOL_SCHEMAS:
        name = schema["function"]["name"]
        assert name in _DISPATCH or name in TERMINAL, name
        assert schema["function"]["description"].strip(), name
