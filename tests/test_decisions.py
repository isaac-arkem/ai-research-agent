"""Typed decisions: the shape of what comes back, read off a faked response."""

import json
import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.services.decisions import MESSAGE_KINDS, Decision, _turn_text, choose, message_kind

ANSWERS = [("yes", "it is"), ("no", "it is not"), ("unclear", "cannot tell")]


def _settings(**kw):
    base = dict(openrouter_api_key="sk-or-test", openai_api_key="sk-test", decision_model="gpt-4o-mini")
    base.update(kw)
    return SimpleNamespace(**base)


def _response(text, top=None):
    """A chat completion whose first token is `text`, with optional top logprobs."""
    logprobs = None
    if top is not None:
        items = [SimpleNamespace(token=t, logprob=math.log(p)) for t, p in top]
        logprobs = SimpleNamespace(content=[SimpleNamespace(token=text, logprob=0.0, top_logprobs=items)])
    choice = SimpleNamespace(message=SimpleNamespace(content=text), logprobs=logprobs)
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(choices=[choice])
    return client


def test_the_probabilities_come_from_the_tokens_not_from_the_model_saying_so():
    client = _response("A", top=[("A", 0.6), ("B", 0.3), (" C", 0.05), ("x", 0.05)])
    with patch("app.services.decisions.OpenAI", return_value=client):
        d = choose("Is it?", ANSWERS, "context", settings=_settings())
    assert d.choice == "yes"
    assert abs(d.confidence - 0.6 / 0.95) < 1e-9          # renormalised over the letters
    assert abs(sum(d.probabilities.values()) - 1) < 1e-9
    assert set(d.probabilities) == {"yes", "no", "unclear"}
    assert abs(d.margin - (0.6 - 0.3) / 0.95) < 1e-9
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["logprobs"] is True and kwargs["temperature"] == 0


def test_without_token_probabilities_the_confidence_is_unknown_not_certain():
    with patch("app.services.decisions.OpenAI", return_value=_response("B")):
        d = choose("Is it?", ANSWERS, "context", settings=_settings())
    assert d.choice == "no" and d.confidence is None and d.probabilities == {}


def test_an_answer_outside_the_options_is_no_answer():
    with patch("app.services.decisions.OpenAI", return_value=_response("Z")):
        d = choose("Is it?", ANSWERS, "context", settings=_settings())
    assert d.choice is None and "not one of the options" in d.error


def test_nothing_raises():
    with patch("app.services.decisions.OpenAI", side_effect=RuntimeError("down")):
        d = choose("Is it?", ANSWERS, "context", settings=_settings())
    assert d.choice is None and "down" in d.error
    assert choose("Is it?", ANSWERS, "c", settings=_settings(openai_api_key=None)).error
    assert choose("Is it?", ANSWERS, "c", settings=_settings(openrouter_api_key=None,
                                                             decision_model="typesafe/jev-router")).error
    assert choose("Is it?", ANSWERS[:1], "c", settings=_settings()).error


def test_gpt_goes_to_openai_and_only_jev_goes_to_openrouter():
    client = _response("A", top=[("A", 1.0)])
    with patch("app.services.decisions.OpenAI", return_value=client) as made:
        choose("Is it?", ANSWERS, "context", settings=_settings(decision_model="gpt-4o-mini"))
    assert "base_url" not in made.call_args.kwargs
    assert made.call_args.kwargs["api_key"] == "sk-test"
    assert client.chat.completions.create.call_args.kwargs["model"] == "gpt-4o-mini"

    with patch("app.services.decisions.OpenAI", return_value=client) as made:
        choose("Is it?", ANSWERS, "context", settings=_settings(decision_model="openai/gpt-4o-mini"))
    assert "base_url" not in made.call_args.kwargs
    assert client.chat.completions.create.call_args.kwargs["model"] == "gpt-4o-mini"

    with patch("app.services.decisions.OpenAI", return_value=client) as made:
        choose("Is it?", ANSWERS, "context", settings=_settings(decision_model="typesafe/jev-router"))
    assert made.call_args.kwargs["base_url"] == "https://openrouter.ai/api/v1"
    assert made.call_args.kwargs["api_key"] == "sk-or-test"
    assert client.chat.completions.create.call_args.kwargs["model"] == "typesafe/jev-router"


def test_assistant_turns_are_read_as_sentences_not_payload():
    stored = json.dumps({"understood_so_far": "Isaac Kusi is the CEO of Krismark.",
                         "clarifying_question": "Search for similar creators?",
                         "creators": [{"name": "Isaac Kusi"}], "web_results": "x" * 5000})
    text = _turn_text("assistant", stored)
    assert "CEO of Krismark" in text and "On screen: Isaac Kusi" in text and "xxxx" not in text


def test_message_kind_offers_exactly_the_four_kinds():
    client = _response("B", top=[("B", 0.9), ("A", 0.1)])
    with patch("app.services.decisions.OpenAI", return_value=client):
        d = message_kind("who is sarkodie?", [SimpleNamespace(role="user", content="find creators like @x")],
                         settings=_settings())
    assert d.choice == "new_question"
    system = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    for label, _ in MESSAGE_KINDS:
        assert label in system


def test_the_decision_is_not_asked_without_a_key_or_when_switched_off():
    from app.services import grounding

    assert grounding._ask_message_kind("hi", [], _settings(openai_api_key=None,
                                                          decision_shadow_enabled=True)) is None
    assert grounding._ask_message_kind("hi", [], _settings(openrouter_api_key=None,
                                                          decision_model="typesafe/jev-router",
                                                          decision_shadow_enabled=True)) is None
    assert grounding._ask_message_kind("hi", [], _settings(decision_shadow_enabled=False,
                                                          decision_steer_enabled=False)) is None


# ── confident decisions steer; anything else changes nothing ─────────


def _job(choice, confidence, error=None):
    from concurrent.futures import Future
    f = Future()
    f.set_result(Decision(choice=choice, confidence=confidence, error=error))
    return f


def test_only_a_confident_decision_steers():
    from app.services.grounding import _message_kind_from

    on = _settings(decision_steer_enabled=True, decision_min_confidence=0.85)
    assert _message_kind_from(_job("new_question", 0.97), {}, "x", on) == "new_question"
    assert _message_kind_from(_job("new_question", 0.60), {}, "x", on) is None
    assert _message_kind_from(_job("new_question", None), {}, "x", on) is None   # unknown confidence
    assert _message_kind_from(_job(None, None, "down"), {}, "x", on) is None
    off = _settings(decision_steer_enabled=False, decision_min_confidence=0.85)
    assert _message_kind_from(_job("new_question", 0.99), {}, "x", off) is None
    assert _message_kind_from(None, {}, "x", on) is None


def _after_a_comparison(kind):
    """"who is sarkodie?" after "creators similar to @khaby.lame on tiktok"."""
    import json as _json
    from app.models.domain import ChatTurn
    from app.services import grounding
    from tests.test_grounding import _ctx, _settings as g_settings, _triage

    history = [ChatTurn(role="user", content="find creators similar to @khaby.lame on tiktok"),
               ChatTurn(role="assistant", content=_json.dumps({"clarifying_question": "x", "missing_fields": []}))]
    client = _triage('{"action":"search","topic":"Sarkodie","answer":"overview",'
                     '"subjects":["Sarkodie"],"reference_accounts":[]}')
    with patch("app.services.grounding.OpenAI", return_value=client), \
         patch("app.services.grounding._ask_message_kind", return_value=None), \
         patch("app.services.grounding._message_kind_from", return_value=kind), \
         patch("app.services.grounding._bases_worth_offering", return_value=[]), \
         patch("app.services.grounding.provider_from_settings"), \
         patch("app.services.grounding._research_via_engine", return_value=None) as engine:
        grounding.gather_web_context("who is sarkodie?", _ctx(), history,
                                     settings=g_settings(research_engine_enabled=True))
    return engine.call_args.kwargs


def test_a_new_question_does_not_inherit_an_earlier_comparison():
    kw = _after_a_comparison("new_question")
    assert kw["seeds"] == [] and kw["subjects"] == ["Sarkodie"]


def test_without_a_confident_decision_the_old_behaviour_stands():
    kw = _after_a_comparison(None)
    assert kw["seeds"] == ["Sarkodie"]          # unchanged: the earlier "similar" still counts


def test_chitchat_with_nothing_to_search_gets_a_reply_and_our_question_again():
    """"alright sure" reached the plan writer: an empty reply, or "Validation
    failed: countries must be a non-empty array"."""
    import json as _json
    from app.models.domain import ChatTurn
    from app.services import agent
    from app.services.grounding import WebContext
    from tests.test_grounding import _ctx, _settings as g_settings

    pending = [ChatTurn(role="assistant", content=_json.dumps({
        "clarifying_question": "Which platform should I look on?", "missing_fields": ["platform"]}))]
    with patch("app.services.agent.gather_web_context",
               return_value=WebContext(action="skip", message_kind="chitchat")), \
         patch("app.services.agent.OpenAI") as planner:
        res = agent.generate_research_plan("alright sure", _ctx(), openai_key="sk",
                                           history=pending, settings=g_settings())
        assert not planner.called
        assert res.ok and res.clarifying_question == "Which platform should I look on?"
        assert res.missing_fields == ["platform"]
        res = agent.generate_research_plan("alright sure", _ctx(), openai_key="sk",
                                           history=[], settings=g_settings())
        assert res.ok and "What would you like to look into next?" in res.understood_so_far


def test_a_skip_that_is_not_chitchat_still_goes_to_the_planner():
    from app.services import agent
    from app.services.grounding import WebContext
    from tests.test_grounding import _ctx, _settings as g_settings

    with patch("app.services.agent.gather_web_context", return_value=WebContext(action="skip")), \
         patch("app.services.agent.OpenAI") as planner:
        planner.return_value.chat.completions.create.side_effect = RuntimeError("planner reached")
        res = agent.generate_research_plan("plan it", _ctx(), openai_key="sk", history=[], settings=g_settings())
    assert planner.return_value.chat.completions.create.called and res.error_code == "openai_failed"


def _response_tokens(text, tokens):
    """A completion whose logprobs cover several tokens: [(token, [(alt, p)])]."""
    content = [SimpleNamespace(token=t, logprob=0.0,
                               top_logprobs=[SimpleNamespace(token=a, logprob=math.log(p)) for a, p in alts])
               for t, alts in tokens]
    choice = SimpleNamespace(message=SimpleNamespace(content=text), logprobs=SimpleNamespace(content=content))
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(choices=[choice])
    return client


def test_a_model_that_thinks_first_is_read_at_the_answer_letter():
    """Jev's routes write "<|sep|>" or "</think>" before the letter."""
    client = _response_tokens("B", [
        ("<|sep|>", [("<|sep|>", 1.0)]),
        (" single", [(" single", 0.8), (" only", 0.2)]),
        ("</think>", [("</think>", 0.9)]),
        ("B", [("B", 0.9), ("A", 0.1)]),
        ("<|close|>", [("<|close|>", 1.0)]),
    ])
    with patch("app.services.decisions.OpenAI", return_value=client):
        d = choose("Is it?", ANSWERS, "context", settings=_settings(decision_model="typesafe/jev-router"))
    assert d.choice == "no" and abs(d.confidence - 0.9) < 1e-9


def test_jev_asks_openrouter_for_providers_that_return_probabilities_and_gpt_does_not():
    client = _response("A", top=[("A", 1.0)])
    with patch("app.services.decisions.OpenAI", return_value=client):
        choose("Is it?", ANSWERS, "context", settings=_settings(decision_model="typesafe/jev-router"))
    assert client.chat.completions.create.call_args.kwargs["extra_body"] == {
        "provider": {"require_parameters": True}, "reasoning": {"effort": "low"}}
    with patch("app.services.decisions.OpenAI", return_value=client):
        choose("Is it?", ANSWERS, "context", settings=_settings(decision_model="gpt-4o-mini"))
    assert "extra_body" not in client.chat.completions.create.call_args.kwargs


def test_a_confident_decline_is_not_searched_even_when_the_router_wants_to():
    """"nope" to "should I look for more creators like Sarkodie?" was routed as
    a search for "who is Sarkodie?" and the same answer came back."""
    from app.services import grounding
    from tests.test_grounding import _ctx, _settings as g_settings, _triage

    client = _triage('{"action":"search","topic":"who is Sarkodie?","subjects":["Sarkodie"]}')
    for kind, searched in (("chitchat", False), (None, True), ("continues_research", True)):
        with patch("app.services.grounding.OpenAI", return_value=client), \
             patch("app.services.grounding._ask_message_kind", return_value=None), \
             patch("app.services.grounding._message_kind_from", return_value=kind), \
             patch("app.services.grounding.provider_from_settings"), \
             patch("app.services.grounding._research_via_engine", return_value=None) as engine:
            web = grounding.gather_web_context("nope", _ctx(), [], settings=g_settings(research_engine_enabled=True))
        assert engine.called is searched, kind
        if kind == "chitchat":
            assert web.action == "skip" and web.message_kind == "chitchat"


# ── the comparison already chosen is remembered ──────────────────────


def _khaby_thread(picked=True):
    import json as _json
    from app.models.domain import ChatTurn

    turns = [ChatTurn(role="user", content="find creators similar to @khaby.lame on tiktok"),
             ChatTurn(role="assistant", content=_json.dumps({
                 "clarifying_question": "Similar in which way?", "missing_fields": ["basis"],
                 "comparison_bases": [{"label": "same humor style"}]}))]
    if picked:
        turns += [ChatTurn(role="user", content="same humor style"),
                  ChatTurn(role="assistant", content=_json.dumps({
                      "clarifying_question": "Search TikTok for more?", "missing_fields": [],
                      "comparison": {"seeds": ["khaby.lame"], "basis": "same humor style"}}))]
    return turns


def _turn(prompt, history, triage_json, kind=None):
    from app.models.domain import ComparisonBasis
    from app.services import grounding
    from app.services.grounding import WebContext
    from tests.test_grounding import _ctx, _settings as g_settings, _triage

    with patch("app.services.grounding.OpenAI", return_value=_triage(triage_json)), \
         patch("app.services.grounding._ask_message_kind", return_value=None), \
         patch("app.services.grounding._message_kind_from", return_value=kind), \
         patch("app.services.grounding._bases_worth_offering",
               return_value=[ComparisonBasis(label="a"), ComparisonBasis(label="b")]) as proposer, \
         patch("app.services.grounding.provider_from_settings"), \
         patch("app.services.grounding._research_via_engine",
               return_value=WebContext(action="search")) as engine:
        web = grounding.gather_web_context(prompt, _ctx(), history,
                                           settings=g_settings(research_engine_enabled=True))
    return web, proposer, engine


_KHABY = ('{"action":"search","topic":"creators similar to @khaby.lame on TikTok",'
          '"answer":"creators","reference_accounts":["khaby.lame"],"platform":"tiktok"}')


def test_the_pick_is_stored_with_the_answer():
    web, proposer, engine = _turn("same humor style", _khaby_thread(picked=False), _KHABY, "continues_research")
    assert not proposer.called
    assert web.comparison == {"seeds": ["khaby.lame"], "basis": "same humor style"}
    # The router already carries the pick in its topic; nothing is appended.
    assert "compared on:" not in engine.call_args.kwargs["prompt"]


def test_more_of_the_same_accounts_keeps_the_basis_already_chosen():
    """"yes search tiktok for more creators like him" was asked "similar in
    which way?" again, right after "same humor style" was picked."""
    web, proposer, engine = _turn("yes search tiktok for more creators like him", _khaby_thread(), _KHABY,
                                  "continues_research")
    assert not proposer.called, "the question is not asked again"
    assert engine.called and web.comparison["basis"] == "same humor style"
    assert "compared on: same humor style" in engine.call_args.kwargs["prompt"]


def test_different_accounts_are_asked_afresh():
    web, proposer, engine = _turn(
        "now find creators similar to @mrbeast on tiktok", _khaby_thread(),
        '{"action":"search","topic":"creators similar to @mrbeast on TikTok","answer":"creators",'
        '"reference_accounts":["mrbeast"],"platform":"tiktok"}', "new_question")
    assert proposer.called and web.missing == ["basis"]


def test_without_a_stored_comparison_it_asks_as_before():
    """An older thread whose answers never stored a comparison: unchanged."""
    import json as _json
    from app.models.domain import ChatTurn

    history = _khaby_thread()[:3] + [ChatTurn(role="assistant", content=_json.dumps({
        "clarifying_question": "Search TikTok for more?", "missing_fields": []}))]
    _, proposer, _ = _turn("yes search tiktok for more creators like him", history, _KHABY, "continues_research")
    assert proposer.called


def test_the_comparison_is_saved_with_the_turn():
    import json
    from app.api.v1.endpoints import research
    from app.models.domain import AgentResult, WebFinding

    saved = []
    store = SimpleNamespace(add_assistant_message=lambda cid, uid, content, **k: saved.append(content) or "m")
    with patch.object(research.conversation_service, "store", store):
        research._persist_assistant("c", "u", AgentResult(
            ok=True, clarifying_question="next?", findings=[WebFinding(title="t", url="https://x", snippet="s")],
            comparison={"seeds": ["khaby.lame"], "basis": "same humor style"}))
    assert json.loads(saved[0])["comparison"] == {"seeds": ["khaby.lame"], "basis": "same humor style"}


# ── the answer to "which platform?" — including "the web only" ───────


def _platform_q():
    import json as _json
    from app.models.domain import ChatTurn
    return [ChatTurn(role="user", content="find creators similar to @khaby.lame"),
            ChatTurn(role="assistant", content=_json.dumps({
                "clarifying_question": "Which platform should I look on — TikTok or Instagram?",
                "missing_fields": ["platform"]}))]


def test_the_platform_decision_is_only_consulted_after_our_platform_question():
    """Asked about a first message, it guessed TikTok for @khaby.lame because
    he is famous there — a guess is not the operator's choice."""
    from app.services import grounding

    s = _settings(decision_steer_enabled=True, decision_min_confidence=0.85)
    with patch("app.services.decisions.choose") as choose_:
        assert grounding._platform_chosen("find creators similar to @khaby.lame", [], s) is None
    assert not choose_.called
    with patch("app.services.decisions.choose", return_value=Decision(choice="web_only", confidence=1.0)):
        assert grounding._platform_chosen("read web pages", _platform_q(), s) == "web_only"


def test_only_a_confident_choice_counts_and_not_said_never_does():
    from app.services import grounding

    s = _settings(decision_steer_enabled=True, decision_min_confidence=0.85)
    for d, want in ((Decision(choice="tiktok", confidence=0.99), "tiktok"),
                    (Decision(choice="tiktok", confidence=0.5), None),
                    (Decision(choice="not_said", confidence=1.0), None),
                    (Decision(choice=None, error="down"), None)):
        with patch("app.services.decisions.choose", return_value=d):
            assert grounding._platform_chosen("x", _platform_q(), s) == want
    off = _settings(decision_steer_enabled=False)
    assert grounding._platform_chosen("read web pages", _platform_q(), off) is None


def _reply_to_platform_question(prompt, choice):
    from app.models.domain import ComparisonBasis
    from app.services import grounding
    from tests.test_grounding import _ctx, _settings as g_settings, _triage

    client = _triage('{"action":"search","topic":"creators similar to @khaby.lame","answer":"creators",'
                     '"reference_accounts":["khaby.lame"],"platform":null}')
    with patch("app.services.grounding.OpenAI", return_value=client), \
         patch("app.services.grounding._ask_message_kind", return_value=None), \
         patch("app.services.grounding._platform_chosen", return_value=choice), \
         patch("app.services.grounding._bases_worth_offering", return_value=[]), \
         patch("app.services.grounding.provider_from_settings"), \
         patch("app.services.grounding._research_via_engine", return_value=None) as engine:
        web = grounding.gather_web_context(prompt, _ctx(), _platform_q(),
                                           settings=g_settings(research_engine_enabled=True))
    return web, engine


def test_web_only_is_an_answer_and_searches_without_a_platform():
    """"read web pages" — the option the question itself offers — was met
    with the same question, forever."""
    web, engine = _reply_to_platform_question("read web pages", "web_only")
    assert web.missing != ["platform"] and engine.called
    assert engine.call_args.kwargs["platform"] is None


def test_a_platform_said_in_other_words_is_used():
    web, engine = _reply_to_platform_question("on the gram", "instagram")
    assert engine.call_args.kwargs["platform"] == "instagram"


def test_no_choice_still_asks_which_platform():
    web, engine = _reply_to_platform_question("alright sure", None)
    assert web.missing == ["platform"] and not engine.called


# ── "more like him" means the account the research is about ──────────


def _more_like(prompt, router_refs, meant):
    import json as _json
    from app.models.domain import ChatTurn
    from app.services import grounding
    from app.services.grounding import WebContext
    from tests.test_grounding import _ctx, _settings as g_settings, _triage

    history = [
        ChatTurn(role="user", content="find creators similar to @khaby.lame on tiktok"),
        ChatTurn(role="assistant", content=_json.dumps({
            "understood_so_far": "CoryXKenshin stands out.", "clarifying_question": "More?",
            "missing_fields": [], "creators": [{"name": "CoryXKenshin", "handle": "coryxkenshin"}],
            "comparison": {"seeds": ["khaby.lame"], "basis": "same humor style"}})),
    ]
    refs = _json.dumps(router_refs)
    client = _triage('{"action":"search","topic":"creators similar to CoryXKenshin on TikTok","answer":"creators",'
                     f'"reference_accounts":{refs},"platform":"tiktok"}}')
    with patch("app.services.grounding.OpenAI", return_value=client), \
         patch("app.services.grounding._ask_message_kind", return_value=None), \
         patch("app.services.grounding._message_kind_from", return_value="continues_research"), \
         patch("app.services.grounding._meant_the_compared_account", return_value=meant) as asked, \
         patch("app.services.grounding._bases_worth_offering", return_value=[]), \
         patch("app.services.grounding.provider_from_settings"), \
         patch("app.services.grounding._research_via_engine", return_value=WebContext(action="search")) as engine:
        grounding.gather_web_context(prompt, _ctx(), history, settings=g_settings(research_engine_enabled=True))
    return asked, engine.call_args.kwargs


def test_him_means_the_compared_account_when_the_decision_says_so():
    """Read as CoryXKenshin five times out of five, while the comparison saved
    with the thread was @khaby.lame."""
    asked, kw = _more_like("find more creators like him", ["coryxkenshin"], True)
    assert asked.called
    assert kw["seeds"] == ["khaby.lame"]
    assert "@khaby.lame" in kw["prompt"] and "coryxkenshin" not in kw["prompt"].lower()
    assert "compared on: same humor style" in kw["prompt"]


def test_unsure_or_someone_in_the_results_keeps_the_routers_reading():
    asked, kw = _more_like("find more like that reaction guy", ["coryxkenshin"], False)
    assert asked.called and kw["seeds"] == ["coryxkenshin"]


def test_a_name_the_operator_typed_is_never_second_guessed():
    asked, kw = _more_like("find more creators like coryxkenshin", ["coryxkenshin"], True)
    assert not asked.called and kw["seeds"] == ["coryxkenshin"]


def test_the_same_account_needs_no_decision():
    asked, kw = _more_like("find more creators like him", ["khaby.lame"], True)
    assert not asked.called and kw["seeds"] == ["khaby.lame"]


def test_a_question_about_someone_on_screen_does_not_inherit_a_comparison():
    """"does he have an instagram account?" after "creators similar to him"
    became a hunt for others like him, with a paid Instagram scrape."""
    kw = _after_a_comparison("about_results")
    assert kw["seeds"] == [] and kw["subjects"] == ["Sarkodie"]


def test_carrying_the_research_on_still_inherits_it():
    kw = _after_a_comparison("continues_research")
    assert kw["seeds"] == ["Sarkodie"]
