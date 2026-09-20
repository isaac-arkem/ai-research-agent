# The engine's reasoning seam, wired to researchAgent's OpenAI client.
#
# The vendored engine asks for a `providers.ReasoningClient` in two places:
# the planner (which we replaced — see orchestrator.plan_for) and
# `rerank.rerank_candidates`. Rerank is the one that matters: rerank_score is
# 60% of a candidate's final score, and with no client it silently falls back
# to a deterministic local heuristic that cannot read the topic.
#
# That fallback is not hypothetical. Every exploratory run of this engine done
# without a client returned candidates scored 0.00 — the floor, running blind.
#
# The engine's own clients (Gemini, OpenAI, xAI, OpenRouter in providers.py)
# each carry their own key handling and endpoint config. We use none of them:
# researchAgent already has an authenticated OpenAI client and one place where
# the model is chosen, and a second credential path for the same vendor is a
# second thing to rotate.

from __future__ import annotations

import logging
from typing import Any, List, Optional

from openai import OpenAI

from app.services.research.engine import providers

logger = logging.getLogger(__name__)


class OpenAIReasoningClient(providers.ReasoningClient):
    """ReasoningClient backed by researchAgent's OpenAI account.

    Only `generate_text` needs implementing — the base class's `generate_json`
    calls it with response_mime_type="application/json" and runs the result
    through `providers.extract_json`, which tolerates the fenced ```json the
    models still emit sometimes.
    """

    name = "openai"

    def __init__(self, api_key: str, *, timeout: float = 60.0) -> None:
        if not api_key:
            # Better here than as an auth error three lanes deep, after the
            # paid Apify actor runs have already been billed.
            raise ValueError("OpenAIReasoningClient requires an API key")
        self._client = OpenAI(api_key=api_key, timeout=timeout)

    def generate_text(
        self,
        model: str,
        prompt: str,
        *,
        tools: Optional[List[dict]] = None,
        response_mime_type: Optional[str] = None,
    ) -> str:
        # `tools` is part of the engine's interface for the Gemini path. We do
        # not pass tools to the reranker, and quietly ignoring them would make
        # a future caller's tools vanish without a word.
        if tools:
            raise NotImplementedError(
                "OpenAIReasoningClient does not forward tools; the rerank and "
                "plan paths do not use them"
            )

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if response_mime_type == "application/json":
            kwargs["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(**kwargs)
        return (response.choices[0].message.content or "").strip()


def build_reasoning_client(settings) -> Optional[OpenAIReasoningClient]:
    """The client, or None when no key is configured.

    None is a supported answer, not a failure: the engine degrades to its
    deterministic fallback. It ranks worse, but it ranks — and a research run
    that returns weaker results beats one that raises.
    """
    key = getattr(settings, "openai_api_key", None)
    if not key:
        logger.warning(
            "no OPENAI_API_KEY: reranking will use the deterministic fallback"
        )
        return None
    return OpenAIReasoningClient(key)
