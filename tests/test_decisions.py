"""Typed decisions: the shape of what comes back, read off a faked response."""

import json
import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.services.decisions import MESSAGE_KINDS, Decision, _turn_text, choose, message_kind

ANSWERS = [("yes", "it is"), ("no", "it is not"), ("unclear", "cannot tell")]


def _settings(**kw):
    base = dict(openrouter_api_key="sk-or-test", decision_model="openai/gpt-4o-mini")
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
    assert choose("Is it?", ANSWERS, "c", settings=_settings(openrouter_api_key=None)).error
    assert choose("Is it?", ANSWERS[:1], "c", settings=_settings()).error


def test_the_call_goes_to_openrouter_with_the_configured_model():
    client = _response("A", top=[("A", 1.0)])
    with patch("app.services.decisions.OpenAI", return_value=client) as made:
        choose("Is it?", ANSWERS, "context", settings=_settings(decision_model="google/gemini-x"))
    assert made.call_args.kwargs["base_url"] == "https://openrouter.ai/api/v1"
    assert client.chat.completions.create.call_args.kwargs["model"] == "google/gemini-x"


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


def test_shadow_mode_is_off_without_a_key_and_never_blocks_the_turn():
    from app.services import grounding

    with patch("threading.Thread") as thread:
        grounding._shadow_decisions("hi", [], {}, _settings(openrouter_api_key=None,
                                                             decision_shadow_enabled=True))
        grounding._shadow_decisions("hi", [], {}, _settings(decision_shadow_enabled=False))
    assert not thread.called
    with patch("threading.Thread") as thread:
        grounding._shadow_decisions("hi", [], {"action": "skip"}, _settings(decision_shadow_enabled=True))
    assert thread.call_args.kwargs["daemon"] is True
    thread.return_value.start.assert_called_once()


def test_a_model_that_writes_around_the_letter_is_still_read():
    """Jev Router answers through a model that may wrap the letter."""
    for said, want in (("B", "no"), ("**B**", "no"), ("Answer: C", "unclear"), (" a ", "yes")):
        with patch("app.services.decisions.OpenAI", return_value=_response(said)):
            d = choose("Is it?", ANSWERS, "context", settings=_settings())
        assert d.choice == want, said
