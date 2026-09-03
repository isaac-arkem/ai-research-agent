"""Agent — sanitise, OpenAI Chat Completions, validate, classify flow."""

import json
import logging
import re
import time
from typing import Optional, Sequence

from openai import OpenAI

from app.models.domain import AgentContext, AgentResult, ChatTurn
from app.services.flows import classify_flow
from app.services.prompt import assemble_system_prompt, build_chat_messages
from app.services.validator import validate_research_plan
from app.utils.guards import sanitize_prompt

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.IGNORECASE).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("No JSON object found in LLM response")
    return json.loads(cleaned[start : end + 1])


def _fail(
    error: str,
    error_code: str,
    *,
    started: float,
    llm_ms: Optional[int] = None,
    raw: Optional[str] = None,
    validation=None,
    plan=None,
) -> AgentResult:
    return AgentResult(
        ok=False,
        error=error,
        error_code=error_code,
        raw=raw,
        validation=validation,
        plan=plan,
        latency_ms=int((time.perf_counter() - started) * 1000),
        llm_latency_ms=llm_ms,
    )


def generate_research_plan(
    operator_prompt: str,
    ctx: AgentContext,
    *,
    openai_key: str,
    model: str = "gpt-4o",
    temperature: float = 0.3,
    max_tokens: int = 4000,
    history: Optional[Sequence[ChatTurn]] = None,
) -> AgentResult:
    started = time.perf_counter()

    sanitized = sanitize_prompt(operator_prompt)
    if not sanitized:
        return _fail("Empty prompt after sanitisation", "empty_prompt", started=started)

    messages = build_chat_messages(
        assemble_system_prompt(ctx),
        sanitized,
        history,
    )

    client = OpenAI(api_key=openai_key)
    llm_ms = None
    prompt_tokens = None
    completion_tokens = None

    try:
        logger.info(
            "Calling OpenAI model=%s prompt_length=%d history_turns=%d",
            model,
            len(sanitized),
            len(history or []),
        )
        llm_started = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        llm_ms = int((time.perf_counter() - llm_started) * 1000)
        raw_text = response.choices[0].message.content or ""
        usage = response.usage
        if usage is not None:
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
        logger.info(
            "LLM responded length=%d llm_latency_ms=%d prompt_tokens=%s",
            len(raw_text),
            llm_ms,
            prompt_tokens,
        )
    except Exception as exc:
        logger.error("OpenAI call failed: %s", exc)
        return _fail(f"OpenAI call failed: {exc}", "openai_failed", started=started)

    try:
        parsed = _extract_json(raw_text)
    except Exception:
        return _fail(
            "Failed to extract JSON from LLM response",
            "json_extract_failed",
            started=started,
            llm_ms=llm_ms,
            raw=raw_text,
        )

    valid_codes = {m.iso for m in ctx.markets}
    validation = validate_research_plan(parsed, valid_codes)
    elapsed = int((time.perf_counter() - started) * 1000)

    if not validation.valid:
        return AgentResult(
            ok=False,
            plan=validation.plan,
            validation=validation,
            raw=raw_text,
            error=f"Validation failed: {'; '.join(e.message for e in validation.errors)}",
            error_code="validation_failed",
            latency_ms=elapsed,
            llm_latency_ms=llm_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    # Clarifying question — valid response but no plan yet
    if "clarifying_question" in parsed:
        return AgentResult(
            ok=True,
            plan=None,
            clarifying_question=parsed.get("clarifying_question"),
            understood_so_far=parsed.get("understood_so_far"),
            missing_fields=parsed.get("missing_fields", []),
            latency_ms=elapsed,
            llm_latency_ms=llm_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    plan = validation.plan
    return AgentResult(
        ok=True,
        plan=plan,
        flow=classify_flow(plan),
        latency_ms=elapsed,
        llm_latency_ms=llm_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
