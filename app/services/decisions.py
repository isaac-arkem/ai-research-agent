"""Typed decisions: one bounded question, a fixed set of answers, a probability
for each.

The router answers many questions at once — action, topic, subjects,
reference accounts, platform, limits — in one free-form JSON object, and the
code after it re-derives several of them from word lists. Most of the
conversation failures traced back to one of those small judgements made badly
and silently: "who is sarkodie?" after a comparison read as another
comparison, "read web pages" not accepted as an answer to "which platform?".

A decision here is the opposite shape. The caller writes down the question and
every allowed answer BEFORE the call; the model can only pick one of them; and
what comes back carries the model's probability for every option, so a caller
can treat a close call as a reason to ask rather than to guess.

The probabilities are read, not requested. Each option is a single letter and
the model is asked for one token; the token probabilities over those letters
ARE the distribution. A model asked to "rate its confidence" reports a number
it made up. When the model returns no token probabilities, `confidence` is
None — reported as unknown, never invented.

Every failure returns a Decision with `choice=None` and the reason in
`error`. Nothing here raises: a decision is advice to the caller, and the
caller must always have a way to carry on without it.
"""

from __future__ import annotations

import json
import logging
import math
import re
import string
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from openai import OpenAI

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _uses_openrouter(model: str) -> bool:
    """OpenRouter is for Jev only. Every GPT model goes to OpenAI directly,
    on the OpenAI key the rest of the app already uses."""
    return "jev" in (model or "").lower()


def _endpoint(model: str, settings) -> tuple:
    """(api key, base url or None, model name as that endpoint knows it)."""
    if _uses_openrouter(model):
        return getattr(settings, "openrouter_api_key", None), OPENROUTER_BASE_URL, model
    # OpenRouter's "openai/gpt-4o-mini" is OpenAI's "gpt-4o-mini".
    name = model.split("/", 1)[1] if model.lower().startswith("openai/") else model
    return getattr(settings, "openai_api_key", None), None, name

# Letters are single tokens for every tokenizer worth using, which is what lets
# the probability of an option be read straight off the first output token.
_LETTERS = string.ascii_uppercase


@dataclass
class Decision:
    """What came back for one question."""

    choice: Optional[str]                     # the label picked, or None on failure
    confidence: Optional[float] = None        # probability of `choice`; None if unknown
    probabilities: Dict[str, float] = field(default_factory=dict)
    model: str = ""
    latency_ms: int = 0
    error: Optional[str] = None

    @property
    def margin(self) -> Optional[float]:
        """Best minus second best. A small margin is a close call even when
        the top probability looks decisive."""
        if len(self.probabilities) < 2:
            return None
        top = sorted(self.probabilities.values(), reverse=True)
        return top[0] - top[1]


def choose(
    question: str,
    answers: Sequence[Tuple[str, str]],
    context: str,
    *,
    settings,
    model: Optional[str] = None,
    timeout: float = 10.0,
) -> Decision:
    """Pick one of `answers` for `question`, given `context`.

    `answers` is a list of (label, what it means). The label is what the
    caller acts on; the meaning is what the model reads. Include a fallback
    answer ("none of these", "unclear") whenever the real world may not fit.
    """
    model = model or getattr(settings, "decision_model", "gpt-4o-mini")
    key, base_url, api_model = _endpoint(model, settings)
    if not key:
        return Decision(choice=None, model=model, error=(
            "no OPENROUTER_API_KEY" if base_url else "no OPENAI_API_KEY"))
    answers = list(answers)
    if not 2 <= len(answers) <= len(_LETTERS):
        return Decision(choice=None, model=model, error=f"{len(answers)} answers")

    letters = _LETTERS[: len(answers)]
    options = "\n".join(
        f"{letter}. {label} — {meaning}" for letter, (label, meaning) in zip(letters, answers)
    )
    system = (
        f"{question}\n\nOptions:\n{options}\n\n"
        f"Answer with the single letter of the best option ({', '.join(letters)}) and nothing else."
    )
    started = time.perf_counter()
    try:
        client = (OpenAI(api_key=key, base_url=base_url, timeout=timeout) if base_url
                  else OpenAI(api_key=key, timeout=timeout))
        response = client.chat.completions.create(
            model=api_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": context},
            ],
            temperature=0,
            # Room to answer, not to write. A model that thinks before it
            # answers — Jev Router sends these to one — spends its first tokens
            # thinking, and with a budget of one it was cut off before the
            # letter. A direct model still answers with the letter first, so
            # its probabilities are still read off that first token.
            max_tokens=int(getattr(settings, "decision_max_tokens", 256)),
            logprobs=True,
            top_logprobs=min(20, max(5, len(answers))),
            # Jev Router forwards each request to a model of its choosing, and
            # its default pick (gemini flash) returns no token probabilities —
            # so every Jev decision came back with no confidence and could
            # never steer. Asking OpenRouter to route only to providers that
            # honour every parameter sent (logprobs included) fixes that.
            # And a short think: given room, a model that reasons first spent
            # all 256 tokens deliberating over "nope" and never answered.
            **({"extra_body": {"provider": {"require_parameters": True},
                               "reasoning": {"effort": "low"}}} if base_url else {}),
        )
    except Exception as exc:
        logger.warning("decision failed (%s): %s", model, exc)
        return Decision(choice=None, model=model, error=f"{type(exc).__name__}: {exc}"[:200],
                        latency_ms=int((time.perf_counter() - started) * 1000))
    latency = int((time.perf_counter() - started) * 1000)

    choice_obj = response.choices[0] if response.choices else None
    raw = ((choice_obj.message.content if choice_obj else "") or "").strip().upper()
    found = re.search(rf"\b([{letters}])\b", raw)
    text = found.group(1) if found else raw[:1]
    by_letter = dict(zip(letters, (label for label, _ in answers)))

    probabilities = _letter_probabilities(choice_obj, letters, answer=text)
    if probabilities:
        best = max(probabilities, key=probabilities.get)
        return Decision(
            choice=by_letter[best],
            confidence=probabilities[best],
            probabilities={by_letter[k]: v for k, v in probabilities.items()},
            model=model, latency_ms=latency,
        )
    if text in by_letter:
        # No token probabilities from this model: the pick stands, its
        # confidence is unknown — not 1.0, which would be a claim.
        return Decision(choice=by_letter[text], model=model, latency_ms=latency)
    return Decision(choice=None, model=model, latency_ms=latency,
                    error=f"answer {text!r} is not one of the options")


def _letter_probabilities(choice_obj, letters: str, answer: str = "") -> Dict[str, float]:
    """The answer token's probability mass over the option letters, normalised.

    The answer token, not the first one. A model that thinks before answering
    — the ones Jev routes to — writes "<|sep|>" or a few words and "</think>"
    first, and the letter several tokens later; the first token's
    probabilities are about that preamble, not the choice. For a model that
    answers straight away the letter IS the first token, so this is the same.
    """
    try:
        tokens = list(choice_obj.logprobs.content or [])
    except Exception:
        return {}
    if not tokens:
        return {}
    picked = tokens[0]
    if answer:
        for token in reversed(tokens):
            if str(getattr(token, "token", "") or "").strip().upper() == answer:
                picked = token
                break
    candidates = list(getattr(picked, "top_logprobs", None) or []) or [picked]
    mass: Dict[str, float] = {}
    for item in candidates:
        token = str(getattr(item, "token", "") or "").strip().upper()
        if token in letters and len(token) == 1:
            mass[token] = mass.get(token, 0.0) + math.exp(getattr(item, "logprob", -math.inf))
    total = sum(mass.values())
    if total <= 0:
        return {}
    return {letter: mass.get(letter, 0.0) / total for letter in letters}


# ── The decisions themselves ────────────────────────────────────────────


MESSAGE_KIND_QUESTION = (
    "You are reading one turn of a conversation with a creator-research "
    "assistant. What is the operator's latest message?"
)

MESSAGE_KINDS: List[Tuple[str, str]] = [
    ("continues_research",
     "carries on the research already under way: answers the assistant's last "
     "question, picks one of its options, names a platform, narrows or widens "
     "the same request, or asks for more of the same"),
    ("new_question",
     "starts something new: a different person, topic or search from the one "
     "under way — including someone who only shares a name with a person on screen"),
    ("about_results",
     "is about the results already on screen: filter, sort, explain or compare "
     "them, or get handles or details for people listed"),
    ("chitchat",
     "a greeting, thanks, acknowledgement, or a decline (\"no\", \"nope\", \"not now\", "
     "\"no thanks\") — nothing to research"),
]


def _turn_text(role: str, content: str) -> str:
    """One turn as a person would read it. Assistant turns are stored as JSON
    for the console; the decision needs the sentences, not the payload."""
    if role != "assistant":
        return f"Operator: {content.strip()}"
    try:
        stored = json.loads(content)
    except ValueError:
        return f"Assistant: {content.strip()[:400]}"
    if not isinstance(stored, dict):
        return f"Assistant: {str(stored)[:400]}"
    parts = [str(stored.get(k) or "").strip() for k in ("understood_so_far", "clarifying_question")]
    names = [str(c.get("name") or c.get("handle") or "") for c in (stored.get("creators") or [])
             if isinstance(c, dict)]
    if names:
        parts.append("On screen: " + ", ".join(n for n in names[:12] if n))
    if not any(parts) and stored.get("summary"):
        parts.append(str(stored.get("summary")))
    return "Assistant: " + " ".join(p for p in parts if p)[:600]


def message_kind(prompt: str, history, *, settings, keep: int = 6) -> Decision:
    """What is this message: more of the same research, something new, a
    question about what is on screen, or chitchat?"""
    turns = []
    for turn in list(history or [])[-keep:]:
        role = getattr(turn, "role", None) or (turn.get("role") if isinstance(turn, dict) else "")
        content = getattr(turn, "content", None) or (turn.get("content") if isinstance(turn, dict) else "")
        turns.append(_turn_text(str(role), str(content or "")))
    context = (
        ("Conversation so far:\n" + "\n".join(turns) + "\n\n" if turns else "No earlier turns.\n\n")
        + f"Latest message: {prompt.strip()}"
    )
    return choose(MESSAGE_KIND_QUESTION, MESSAGE_KINDS, context, settings=settings)
