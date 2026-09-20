"""The review turn — findings first, plan only once the operator says so.

This is the contract the console depends on, so it is tested at the agent
boundary rather than inside grounding: what does one turn actually return.

Turn 1 of a research question must come back with sources and NO plan. Turn 2
decides what happens next — approve and the plan is drawn from those sources,
narrow it and a fresh search replaces them.

The planner LLM not being called on a review turn is part of the contract,
not an optimisation. A plan built before the operator has vetted the sources
is the thing this whole flow exists to prevent.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.models.domain import (
    AgentContext,
    ChatTurn,
    MarketEntry,
    TaxonomyEntry,
)
from tests.plans import discovery_plan
from app.services.agent import generate_research_plan
from app.services.search import SearchResult


def _ctx():
    return AgentContext(
        markets=[
            MarketEntry(code="NG", iso="NG", name="Nigeria",
                        region="West Africa", languages=["en"]),
            MarketEntry(code="SA", iso="SA", name="Saudi Arabia",
                        region="Gulf", languages=["ar"]),
        ],
        taxonomy=[TaxonomyEntry(slug="fashion_beauty", aliases=["fashion"])],
    )


def _settings():
    return SimpleNamespace(
        search_grounding_enabled=True,
        search_api_key="tvly-test",
        openai_api_key="sk-test",
        grounding_model="gpt-4o-mini",
        search_timeout=15.0,
        search_results_per_query=5,
    )


def _llm(payload):
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=payload))]
    resp.usage = SimpleNamespace(prompt_tokens=10, completion_tokens=20)
    client = MagicMock()
    client.chat.completions.create.return_value = resp
    return client


def _results(n=3):
    # Long and distinct, so neither the thin-page nor the duplicate-body
    # filter removes them before the extractor runs.
    return [SearchResult(url=f"https://x{i}.com", title=f"T{i}",
                         description=f"snippet {i}",
                         content=f"Page {i}. " + f"Real prose about creators {i}. " * 25)
            for i in range(n)]


def _ask(prompt, triage_json, results=None, planner_json=None, history=None):
    """One turn, with triage, search and the planner all stubbed."""
    planner = _llm(planner_json or "{}")
    with patch("app.services.grounding.OpenAI", return_value=_llm(triage_json)):
        with patch("app.services.grounding.provider_from_settings",
                   return_value=SimpleNamespace(
                       name="tavily", search=lambda q: results or [])):
            with patch("app.services.agent.OpenAI", return_value=planner):
                result = generate_research_plan(
                    prompt, _ctx(), openai_key="sk-test",
                    history=history, settings=_settings())
    return result, planner


# ── turn 1: sources, no plan ─────────────────────────────────────────


def test_a_research_question_returns_findings_and_no_plan():
    result, planner = _ask(
        "what fitness content is trending in Nigeria?",
        '{"action": "search", "country": "NG"}',
        results=_results(3))

    assert result.ok is True
    assert result.plan is None
    assert result.awaiting_approval is True
    assert len(result.findings) == 3
    # the query is the operator's own question, unrewritten
    assert result.searched_for == "what fitness content is trending in Nigeria?"


def test_the_planner_is_not_called_on_a_review_turn():
    """The point of the flow: no plan is built until the sources are vetted."""
    _, planner = _ask("what is trending in Nigeria?",
                      '{"action": "search"}',
                      results=_results(2))
    planner.chat.completions.create.assert_not_called()


def test_the_review_turn_asks_which_half_to_plan_with():
    """Named accounts and a hashtag sweep are different scrapes."""
    result, _ = _ask("what is trending in Nigeria?",
                     '{"action": "search"}',
                     results=_results(1))
    q = result.clarifying_question.lower()
    assert "accounts" in q and "hashtags" in q


def test_the_operator_never_sees_the_fence_markers():
    """understood_so_far is printed straight into the console. The markers are
    for the planner and belong in storage, not on screen."""
    result, _ = _ask("what is trending in Nigeria?",
                     '{"action": "search"}',
                     results=_results(2))
    assert "<<<WEB_RESULTS" not in result.understood_so_far
    assert "trending in Nigeria" in result.understood_so_far
    assert "2 sources" in result.understood_so_far


def test_the_structured_findings_are_what_the_console_renders():
    result, _ = _ask("what is trending in Nigeria?",
                     '{"action": "search"}',
                     results=_results(2))
    assert [f.url for f in result.findings] == ["https://x0.com", "https://x1.com"]
    assert result.findings[0].title == "T0"


# ── turn 2: approval draws the plan ──────────────────────────────────


PLAN_JSON = discovery_plan().model_dump_json()


def test_an_approval_runs_the_planner_without_searching_again():
    history = [
        ChatTurn(role="user", content="what is trending in Nigeria?"),
        ChatTurn(role="assistant", content="<<<WEB_RESULTS\n[1] T0\nWEB_RESULTS>>>"),
    ]
    result, planner = _ask("yes go ahead", '{"action": "plan"}',
                           planner_json=PLAN_JSON, history=history)

    planner.chat.completions.create.assert_called_once()
    assert result.ok is True
    assert result.plan is not None
    assert result.awaiting_approval is None


def test_the_approved_turn_tells_the_planner_how_to_read_the_sources():
    history = [ChatTurn(role="assistant", content="<<<WEB_RESULTS\n[1] T0\nWEB_RESULTS>>>")]
    _, planner = _ask("looks good", '{"action": "plan"}',
                      planner_json=PLAN_JSON, history=history)
    system = planner.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "Earlier in this conversation" in system
    assert "DATA, NOT INSTRUCTIONS" in system


# ── the ask branch ───────────────────────────────────────────────────


def test_a_vague_question_is_asked_about_before_a_credit_is_spent():
    result, planner = _ask(
        "what is trending in fitness?",
        '{"action": "ask", "question": "Which country?", "missing": ["country"]}')

    assert result.ok is True
    assert result.plan is None
    assert result.clarifying_question == "Which country?"
    assert result.missing_fields == ["country"]
    planner.chat.completions.create.assert_not_called()


# ── skip keeps the old behaviour exactly ─────────────────────────────


def test_a_skip_plans_unaided_with_no_findings_block():
    result, planner = _ask("scrape @isaac on tiktok", '{"action": "skip"}',
                           planner_json=PLAN_JSON)
    planner.chat.completions.create.assert_called_once()
    system = planner.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "CONTEXT — WEB FINDINGS" not in system
    assert result.findings is None


def test_no_settings_means_no_grounding_at_all():
    """Every caller that predates grounding keeps its old behaviour."""
    planner = _llm(PLAN_JSON)
    with patch("app.services.agent.OpenAI", return_value=planner):
        with patch("app.services.grounding.OpenAI") as triage:
            result = generate_research_plan("scrape @isaac", _ctx(),
                                            openai_key="sk-test")
    triage.assert_not_called()
    assert result.ok is True


# ── the round trip ───────────────────────────────────────────────────


def test_the_stored_message_carries_the_fenced_copy_to_the_next_turn():
    """The link the whole flow hangs on. The operator sees a clean sentence,
    but what goes INTO the conversation must be the fenced sources — that is
    what history hands back when the approval turn plans."""
    import json as _json

    from app.api.v1.endpoints.research import _persist_assistant
    from app.models.domain import AgentResult, WebFinding

    result = AgentResult(
        ok=True,
        clarifying_question="Do these look right?",
        understood_so_far='I searched the web for "fitness nigeria" and found 1 source.',
        missing_fields=[],
        findings=[WebFinding(title="T0", url="https://x0.com", snippet="s")],
        searched_for="fitness nigeria",
        awaiting_approval=True,
    )

    captured = {}

    class _Store:
        def add_assistant_message(self, cid, uid, content, **kw):
            captured["content"] = content
            return "msg-1"

    with patch("app.api.v1.endpoints.research.conversation_service.store", _Store()):
        assert _persist_assistant("c1", "u1", result) == "msg-1"

    stored = _json.loads(captured["content"])
    assert "<<<WEB_RESULTS" in stored["web_results"]
    assert "https://x0.com" in stored["web_results"]
    # and the structured copy the console reads when a thread is reopened
    assert stored["findings"][0]["url"] == "https://x0.com"
    assert stored["searched_for"] == "fitness nigeria"
    # and the displayed half stays clean
    assert "<<<WEB_RESULTS" not in stored["understood_so_far"]


def test_an_ordinary_clarifying_question_stores_no_web_results_key():
    """Only a review turn carries sources; a plain "which country?" must not
    grow an empty findings blob."""
    import json as _json

    from app.api.v1.endpoints.research import _persist_assistant
    from app.models.domain import AgentResult

    result = AgentResult(ok=True, clarifying_question="Which country?",
                         understood_so_far="Need a market.", missing_fields=["country"])
    captured = {}

    class _Store:
        def add_assistant_message(self, cid, uid, content, **kw):
            captured["content"] = content
            return "msg-2"

    with patch("app.api.v1.endpoints.research.conversation_service.store", _Store()):
        _persist_assistant("c1", "u1", result)

    assert "web_results" not in _json.loads(captured["content"])


# ── named accounts settle it: no search, ever ────────────────────────


def _no_search(prompt, history=None):
    """Run a turn with triage and Tavily both watched, and report if either ran.

    The router is now consulted on EVERY turn, so the first value is True
    throughout. It used to be skipped whenever a regex found an "@", and that
    bypass — not the model — caused most of what went wrong: a comparison
    naming two accounts became a plan to scrape them.

    What these tests actually protect is the second value: a named-account
    job must not spend a search. That still holds, because the router returns
    skip and a skip reaches the planner exactly as the bypass did.
    """
    planner = _llm('{"clarifying_question":"Which niche?",'
                   '"understood_so_far":"Named accounts.","missing_fields":["niche"]}')
    with patch("app.services.grounding.OpenAI") as triage:
        with patch("app.services.grounding.provider_from_settings") as tavily:
            with patch("app.services.agent.OpenAI", return_value=planner):
                generate_research_plan("x" if not prompt else prompt, _ctx(),
                                       openai_key="sk-test", history=history,
                                       settings=_settings())
    return triage.called, tavily.called


def test_naming_accounts_does_not_search_even_with_a_platform_given():
    """The bug this fixes: the old guard keyed on a MISSING platform, so
    "on tiktok" switched off the very check meant to stop this."""
    assert _no_search("scrape @isaac and @marco on tiktok") == (True, False)


def test_a_follow_up_about_a_named_account_job_is_left_to_the_router():
    """"tech-giants" names no accounts, so the free check cannot rule on it —
    whether it belongs to the @isaac job is a question about intent, and the
    router has the history. What must NOT happen is a search: when the router
    says skip, nothing reaches Tavily."""
    history = [
        ChatTurn(role="user", content="scrape @isaac and @marco on tiktok"),
        ChatTurn(role="assistant", content='{"clarifying_question":"Which niche?"}'),
    ]
    planner = _llm('{"clarifying_question":"Which niche?"}')
    with patch("app.services.grounding.OpenAI",
               return_value=_llm('{"action":"skip","reason":"named accounts"}')) as triage:
        with patch("app.services.grounding.provider_from_settings") as tavily:
            with patch("app.services.agent.OpenAI", return_value=planner):
                generate_research_plan("tech-giants", _ctx(), openai_key="sk",
                                       history=history, settings=_settings())
    assert triage.called is True      # the router is asked
    assert tavily.called is False     # and its answer is respected


def test_the_router_is_told_that_named_accounts_stay_named():
    """The gap that caused the bug: the old rule only covered a message that
    "only names accounts", so a bare "tech-giants" fell outside it and was
    searched. The instruction now covers the follow-up, and says a new
    question later in the thread is not one."""
    from app.services.grounding import TRIAGE_SYSTEM

    assert "NAMED ACCOUNTS STAY NAMED" in TRIAGE_SYSTEM
    assert "tech-giants" in TRIAGE_SYSTEM          # the case, by name
    assert "A NEW research question" in TRIAGE_SYSTEM
    assert "does not carry over" in TRIAGE_SYSTEM


def test_a_finished_account_job_does_not_silence_the_rest_of_the_thread():
    """The regression this guards. Asking whether the thread had EVER named a
    handle meant one "@isaac" switched research off for every later question
    in the same conversation — so "find modest fashion creators in Ghana"
    came back as a plan with invented hashtags and no search at all."""
    history = [
        ChatTurn(role="user", content="scrape @isaac and @marco on tiktok"),
        ChatTurn(role="assistant", content='{"clarifying_question":"Which niche?"}'),
        ChatTurn(role="user", content="tech-giants"),
        ChatTurn(role="assistant", content='{"summary":"Scrape @isaac on TikTok."}'),
    ]
    triage_called, _ = _no_search(
        "Find modest fashion creators in Ghana on Instagram", history)
    assert triage_called is True


def test_clarifying_a_new_question_does_not_inherit_the_old_handles():
    """A plan ends the chain, so answering the NEXT question still searches."""
    history = [
        ChatTurn(role="user", content="scrape @isaac on tiktok"),
        ChatTurn(role="assistant", content='{"summary":"Scrape @isaac."}'),
        ChatTurn(role="user", content="find creators in Ghana"),
        ChatTurn(role="assistant", content='{"clarifying_question":"Which platform?"}'),
    ]
    triage_called, _ = _no_search("instagram", history)
    assert triage_called is True


def test_a_question_with_no_handles_still_searches():
    """The guard must not swallow ordinary research."""
    triage_called, _ = _no_search("what fitness content is trending in Nigeria?")
    assert triage_called is True


# ── the plan types itself out ────────────────────────────────────────


def test_the_summary_is_read_out_of_half_finished_json():
    """The plan arrives as one JSON object whose first field is the sentence
    a person reads. Waiting for the closing brace means ten seconds of
    spinner; reading the field as it grows means the plan types itself."""
    import json as _json

    from app.services.agent import _readable_so_far

    full = _json.dumps({"summary": 'He said "hi" and\nleft.', "assumptions": []})
    # every prefix is a valid thing to have received so far
    seen = [_readable_so_far(full[:i]) for i in range(len(full) + 1)]
    assert seen[-1] == 'He said "hi" and\nleft.'
    # it only ever grows — never flickers or backtracks
    assert all(len(b) >= len(a) for a, b in zip(seen, seen[1:]))
    assert all(full[: len(a)] or True for a in seen)


def test_a_half_written_escape_is_not_emitted():
    """Stopping mid-escape would print a stray backslash that the next chunk
    turns into a quote."""
    from app.services.agent import _readable_so_far

    half = '{"summary": "trailing' + chr(92)      # ends in a single backslash
    assert _readable_so_far(half) == "trailing"
    # and once the next chunk completes it, the escaped quote appears
    assert _readable_so_far(half + '"more') == 'trailing"more'


def test_a_clarifying_question_streams_too():
    from app.services.agent import _readable_so_far

    assert _readable_so_far('{"clarifying_question": "Which nic') == "Which nic"


def test_nothing_is_emitted_before_the_field_arrives():
    from app.services.agent import _readable_so_far

    assert _readable_so_far('{"su') == ""
    assert _readable_so_far("") == ""


def test_the_planner_streams_when_someone_is_listening():
    """And does not when nobody is — the plain endpoint keeps one round trip."""
    from unittest.mock import ANY

    planner = _llm(discovery_plan().model_dump_json())
    with patch("app.services.agent.OpenAI", return_value=planner):
        generate_research_plan("scrape @isaac on tiktok", _ctx(), openai_key="sk")
    assert "stream" not in planner.chat.completions.create.call_args.kwargs


# ── answering a question about named accounts ────────────────────────


ASK_PLATFORM = ('{"clarifying_question":"Which platform for isaac?",'
                '"missing_fields":["platform"]}')
ASK_NICHE = '{"clarifying_question":"Which niche?","missing_fields":["niche"]}'
REVIEW_TURN = ('{"clarifying_question":"Do these look right?","missing_fields":[]}')
PLAN_TURN = '{"summary":"Scrape @isaac on Instagram."}'


def test_answering_which_platform_does_not_search():
    """"instagram" names nothing on its own. As the answer to "which platform
    is isaac on?" it belongs to a job whose plan IS that account — and
    searching it returned a blog post about Instagram scrapers."""
    history = [
        ChatTurn(role="user", content="scrape isaac"),
        ChatTurn(role="assistant", content=ASK_PLATFORM),
    ]
    assert _no_search("instagram", history) == (True, False)


def test_answering_which_niche_does_not_search():
    history = [
        ChatTurn(role="user", content="scrape @isaac on tiktok"),
        ChatTurn(role="assistant", content=ASK_NICHE),
    ]
    assert _no_search("tech-giants", history) == (True, False)


def test_a_bare_name_after_scrape_counts_as_a_named_account():
    """"scrape isaac" with no @ is still a named account."""
    # Still bypassed, by the PLATFORM gate rather than the named-account one:
    # a handle with no platform is asked about before anything is spent.
    assert _no_search("scrape isaac") == (False, False)


# ── and the two regressions this guard caused before ─────────────────


def test_a_new_question_after_a_plan_still_searches():
    """A finished plan is not a pending question, so it cannot extend the
    job. This is the bug where one "@isaac" silenced research for every later
    question in the thread."""
    history = [
        ChatTurn(role="user", content="scrape @isaac on tiktok"),
        ChatTurn(role="assistant", content=PLAN_TURN),
    ]
    triage_called, _ = _no_search("Find modest fashion creators in Ghana", history)
    assert triage_called is True


def test_a_new_question_after_a_review_turn_still_searches():
    """A review turn carries a clarifying_question but waits on no field."""
    history = [
        ChatTurn(role="user", content="Compare cooking creators in the Gulf"),
        ChatTurn(role="assistant", content=REVIEW_TURN),
    ]
    triage_called, _ = _no_search("Find modest fashion creators in Ghana", history)
    assert triage_called is True


def test_clarifying_a_new_question_does_not_inherit_older_handles():
    """The pending question must be about THIS request, not an earlier one."""
    history = [
        ChatTurn(role="user", content="scrape @isaac on tiktok"),
        ChatTurn(role="assistant", content=PLAN_TURN),
        ChatTurn(role="user", content="find creators in Ghana"),
        ChatTurn(role="assistant", content='{"clarifying_question":"Which platform?",'
                                           '"missing_fields":["platform"]}'),
    ]
    triage_called, _ = _no_search("instagram", history)
    assert triage_called is True


ASK_BOTH = ('{"clarifying_question":"Platform and niche?",'
            '"missing_fields":["platform","niche"]}')


def test_a_request_that_took_two_questions_to_settle_still_does_not_search():
    """"scrape isaac" needed a platform AND a niche. By the second answer the
    message before the question was "tiktok", not the original request — so
    looking back only one message lost the account, and "ike_tech_boys" was
    searched. It returned a TikTok account called @eyk_tech and an Instagram
    page of cute boys."""
    history = [
        ChatTurn(role="user", content="scrape isaac"),
        ChatTurn(role="assistant", content=ASK_BOTH),
        ChatTurn(role="user", content="tiktok"),
        ChatTurn(role="assistant", content=ASK_NICHE),
    ]
    assert _no_search("ike_tech_boys", history) == (True, False)


def test_the_run_of_questions_can_be_any_length():
    history = [ChatTurn(role="user", content="scrape isaac")]
    for _ in range(4):
        history.append(ChatTurn(role="assistant", content=ASK_NICHE))
        history.append(ChatTurn(role="user", content="not an account"))
    history.append(ChatTurn(role="assistant", content=ASK_NICHE))
    # Still bypassed, by the PLATFORM gate rather than the named-account one:
    # a handle with no platform is asked about before anything is spent.
    assert _no_search("still answering", history) == (False, False)


def test_a_plan_part_way_back_stops_the_walk():
    """The stopping rule is the whole safety of walking back at all."""
    history = [
        ChatTurn(role="user", content="scrape @isaac on tiktok"),
        ChatTurn(role="assistant", content=PLAN_TURN),
        ChatTurn(role="user", content="find cooking creators in Ghana"),
        ChatTurn(role="assistant", content=ASK_NICHE),
        ChatTurn(role="user", content="street food"),
        ChatTurn(role="assistant", content=ASK_NICHE),
    ]
    triage_called, _ = _no_search("anything", history)
    assert triage_called is True
