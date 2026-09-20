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


def test_reading_a_named_account_does_not_need_a_named_platform():
    """Sweeping a hashtag is a fishing trip on a platform the operator chose;
    reading @janedoe is looking at an account they named themselves. Asking
    them to also name the platform would demand something more general than
    what they already gave."""
    tc = _tc("creators like @janedoe")          # no platform anywhere
    with patch("app.services.tools.orchestrator_config",
               return_value={"APIFY_API_TOKEN": "apify-test"}), \
         patch("app.services.research.engine.apify_social.search_tiktok_apify",
               return_value={"items": [
                   {"text": "a post", "author_name": "janedoe", "author_fans": 1000}
               ]}) as actor:
        out = run_tool("profile", {"handle": "@janedoe", "platform": "tiktok"}, tc)

    assert actor.called
    assert tc.paid_calls == 1
    assert "1,000 followers" in out
    # ...but a hashtag sweep on that same turn is still refused.
    assert "refused" in run_tool(
        "search_social", {"platform": "tiktok", "query": "x"}, tc)


def test_profile_respects_the_same_budget():
    tc = _tc("creators like @janedoe", tools_max_paid_calls=1)
    with patch("app.services.tools.orchestrator_config",
               return_value={"APIFY_API_TOKEN": "apify-test"}), \
         patch("app.services.research.engine.apify_social.search_tiktok_apify",
               return_value={"items": [{"text": "x", "author_name": "a"}]}) as actor:
        run_tool("profile", {"handle": "a", "platform": "tiktok"}, tc)
        out = run_tool("profile", {"handle": "b", "platform": "tiktok"}, tc)
    assert actor.call_count == 1
    assert "budget for this turn is spent" in out


def test_an_empty_profile_says_why_rather_than_pretending():
    tc = _tc("creators like @ghost")
    with patch("app.services.tools.orchestrator_config",
               return_value={"APIFY_API_TOKEN": "apify-test"}), \
         patch("app.services.research.engine.apify_social.search_tiktok_apify",
               return_value={"items": []}):
        out = run_tool("profile", {"handle": "ghost", "platform": "tiktok"}, tc)
    assert "no posts" in out and ("private" in out or "misspelled" in out)


def test_a_profile_reads_only_the_account_it_asked_for():
    """The follower count has to come from THEIR post.

    @sarkodie read 147,100 followers one minute and 9,214 the next, because
    the lane was handed the handle as a keyword search as well as a profile,
    and the count was taken off whichever item sorted first — usually a fan
    page. Two different people, both reported as him."""
    tc = _tc("creators like @sarkodie")
    items = [
        {"text": "sarkodie is the goat", "author_name": "sarkodiefanpage",
         "author_fans": 9214, "hashtags": ["sarkodiefanpage"]},
        {"text": "new record out friday", "author_name": "sarkodie",
         "author_fans": 147100, "hashtags": ["ghanamusic"]},
    ]
    with patch("app.services.tools.orchestrator_config",
               return_value={"APIFY_API_TOKEN": "apify-test"}), \
         patch("app.services.research.engine.apify_social.search_tiktok_apify",
               return_value={"items": items}):
        out = run_tool("profile", {"handle": "@sarkodie", "platform": "tiktok"}, tc)

    assert "147,100 followers" in out
    assert "9,214" not in out
    assert "1 recent posts" in out
    assert "#sarkodiefanpage" not in out          # the fan page's tag, not his
    assert "#ghanamusic" in out
    assert "new record out friday" in out
    assert "sarkodie is the goat" not in out


def test_a_profile_asks_the_actor_for_the_profile_and_not_the_name():
    """Passing the handle as the topic made the actor run a keyword sweep
    alongside the profile read, which is what mixed the accounts together —
    and on Instagram it also paid for a hashtag run nobody asked for."""
    tc = _tc("creators like @sarkodie")
    with patch("app.services.tools.orchestrator_config",
               return_value={"APIFY_API_TOKEN": "apify-test"}), \
         patch("app.services.research.engine.apify_social.search_tiktok_apify",
               return_value={"items": []}) as actor:
        run_tool("profile", {"handle": "@sarkodie", "platform": "tiktok"}, tc)

    topic = actor.call_args.args[0]
    assert not topic, f"the lane was given a topic to search: {topic!r}"
    assert actor.call_args.kwargs["creators"] == ["sarkodie"]


def test_a_profile_that_returns_only_strangers_says_so():
    """Silence beats a confident wrong number: if the account is not in what
    came back, the tool must not read a stranger's followers instead."""
    tc = _tc("creators like @ghost")
    with patch("app.services.tools.orchestrator_config",
               return_value={"APIFY_API_TOKEN": "apify-test"}), \
         patch("app.services.research.engine.apify_social.search_tiktok_apify",
               return_value={"items": [
                   {"text": "hi", "author_name": "someoneelse", "author_fans": 500}
               ]}):
        out = run_tool("profile", {"handle": "ghost", "platform": "tiktok"}, tc)

    assert "no posts of their own" in out
    assert "@someoneelse" in out
    assert "500" not in out


def test_a_profile_matches_the_handle_however_it_is_written():
    tc = _tc("creators like @JaneDoe")
    with patch("app.services.tools.orchestrator_config",
               return_value={"APIFY_API_TOKEN": "apify-test"}), \
         patch("app.services.research.engine.apify_social.search_instagram_apify",
               return_value={"items": [
                   {"text": "post", "author_name": "@JANEDOE", "author_fans": 4200}
               ]}):
        out = run_tool("profile", {"handle": "@JaneDoe", "platform": "instagram"}, tc)

    assert "4,200 followers" in out


def test_a_profile_says_who_the_handle_belongs_to():
    """tiktok.com/@sarkodie is a real account with 30 followers whose display
    name is "comfortagyeiwaa46" — not the musician. A stable number is not the
    same as the right person, so the reply has to carry what tells them
    apart: the display name, and whether the account is verified."""
    tc = _tc("read @sarkodie")
    with patch("app.services.tools.orchestrator_config",
               return_value={"APIFY_API_TOKEN": "apify-test"}), \
         patch("app.services.research.engine.apify_social.search_tiktok_apify",
               return_value={"items": [{
                   "text": "clip", "author_name": "sarkodie", "author_fans": 30,
                   "author_nickname": "comfortagyeiwaa46", "author_verified": False,
               }]}):
        out = run_tool("profile", {"handle": "@sarkodie", "platform": "tiktok"}, tc)

    assert 'display name "comfortagyeiwaa46"' in out
    assert "not verified" in out
    assert "30 followers" in out


def test_a_profile_does_not_repeat_the_handle_as_a_display_name():
    tc = _tc("read @janedoe")
    with patch("app.services.tools.orchestrator_config",
               return_value={"APIFY_API_TOKEN": "apify-test"}), \
         patch("app.services.research.engine.apify_social.search_tiktok_apify",
               return_value={"items": [{
                   "text": "clip", "author_name": "janedoe", "author_fans": 900000,
                   "author_nickname": "JaneDoe", "author_verified": True,
               }]}):
        out = run_tool("profile", {"handle": "@janedoe", "platform": "tiktok"}, tc)

    assert "display name" not in out
    assert out.count("verified") == 1 and "not verified" not in out
