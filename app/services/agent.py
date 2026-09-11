"""Agent — sanitise, OpenAI Chat Completions, validate, classify flow."""

import json
import logging
import re
import time
from typing import Optional, Sequence

from openai import OpenAI

from app.models.domain import AgentContext, AgentResult, ChatTurn
from app.services.clamp import clamp_parameters
from app.services.country_filter import filter_unsupported_countries
from app.services.flows import classify_flow
from app.services.grounding import (
    REVIEW_QUESTION,
    gather_web_context,
    summarise_findings,
)
from app.services.known_accounts import (
    handles_needing_platform,
    known_accounts_from_text,
    continues_named_account_job,
    names_accounts,
    platform_clarifying_question,
)
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


# The plan comes back as one JSON object, and its first field is the one
# sentence a person actually reads. So while the model writes, we pull that
# field out of the half-finished JSON and send it on as it grows — the plan
# types itself out instead of the console showing a spinner for ten seconds.
#
# Everything after that field is structure: runs, hashtags, counts. There is
# nothing to type out there, and it renders as cards the moment the object
# closes.
_READABLE = re.compile(r'"(?:summary|clarifying_question)"\s*:\s*"')
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "/": "/"}


def _readable_so_far(raw: str) -> str:
    """The first human-readable string in a partly-received JSON object.

    Written for text that is still arriving: an unterminated string returns
    what has been received, and a half-written escape stops cleanly rather
    than emitting a stray backslash that the next chunk would complete.
    """
    match = _READABLE.search(raw)
    if not match:
        return ""
    out = []
    i = match.end()
    while i < len(raw):
        char = raw[i]
        if char == "\\":
            if i + 1 >= len(raw):
                break  # the escape is still in flight
            out.append(_ESCAPES.get(raw[i + 1], raw[i + 1]))
            i += 2
            continue
        if char == '"':
            break  # the string closed
        out.append(char)
        i += 1
    return "".join(out)


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
    settings=None,
    on_progress=None,
) -> AgentResult:
    started = time.perf_counter()

    sanitized = sanitize_prompt(operator_prompt)
    if not sanitized:
        return _fail("Empty prompt after sanitisation", "empty_prompt", started=started)

    known = known_accounts_from_text(sanitized, history)
    missing_platforms = handles_needing_platform(sanitized, history, known)
    # Named accounts settle the question the search would be asking. Once the
    # operator has said "scrape @isaac and @marco", the plan IS those two, and
    # a web search can only turn up different people with similar names — as
    # it did, offering three strangers called Isaac against a request for one.
    #
    # Two checks, both narrow. This message naming accounts is unambiguous.
    # So is answering a question we just asked ABOUT accounts already named:
    # "instagram" names nothing on its own, but as the reply to "which
    # platform is 'isaac' on?" it belongs to a job whose plan IS that account.
    # Both are scoped to one exchange, so neither can silence research for the
    # rest of the conversation — an earlier version did exactly that.
    named_account_turn = names_accounts(sanitized) or continues_named_account_job(
        sanitized, history
    )

    # Route the turn before planning it. A research question goes to the web
    # first and comes back as findings for the operator to approve; only the
    # turn after that draws a plan. Anything grounding cannot help with —
    # including every way grounding can fail — falls through to the planner
    # exactly as it ran before this existed.
    web = None
    if settings is not None and not missing_platforms and not named_account_turn:
        web = gather_web_context(
            sanitized, ctx, history, settings=settings, on_progress=on_progress
        )

        if web.action == "ask":
            # Missing something the search needs. Ask before spending the
            # credit, not after returning five useless sources.
            return AgentResult(
                ok=True,
                plan=None,
                clarifying_question=web.question,
                understood_so_far=(
                    "I can search the web for this, but I need one more detail first."
                ),
                missing_fields=web.missing,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )

        if web.action == "search":
            # The review turn: sources, no plan. understood_so_far is the
            # sentence the console prints, and `findings` is what it renders
            # as cards. The fenced copy the planner reads next turn is written
            # into the conversation by the endpoint, not shown here — the
            # operator should never see the markers.
            return AgentResult(
                ok=True,
                plan=None,
                clarifying_question=REVIEW_QUESTION,
                understood_so_far=summarise_findings(web),
                missing_fields=[],
                findings=web.findings,
                creators=web.creators,
                hashtags=web.hashtags,
                searched_for=web.query,
                awaiting_approval=True,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )

    messages = build_chat_messages(
        assemble_system_prompt(
            ctx,
            known_accounts=known,
            handles_needing_platform=missing_platforms,
            web=web,
        ),
        sanitized,
        history,
    )

    if on_progress:
        on_progress("planning", detail="Building the plan")

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
        common = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            
        )
        usage = None
        if on_progress is None:
            response = client.chat.completions.create(**common)
            raw_text = response.choices[0].message.content or ""
            usage = response.usage
        else:
            # Same request, read as it arrives.
            raw_text = ""
            sent = 0
            stream = client.chat.completions.create(
                **common, stream=True, stream_options={"include_usage": True}
            )
            for chunk in stream:
                if getattr(chunk, "usage", None) is not None:
                    usage = chunk.usage
                if not chunk.choices:
                    continue
                piece = chunk.choices[0].delta.content
                if not piece:
                    continue
                raw_text += piece
                readable = _readable_so_far(raw_text)
                if len(readable) > sent:
                    on_progress("writing", text=readable[sent:])
                    sent = len(readable)
        llm_ms = int((time.perf_counter() - llm_started) * 1000)
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

    elapsed = int((time.perf_counter() - started) * 1000)

    if missing_platforms:
        asked = parsed.get("clarifying_question")
        if not isinstance(asked, str) or "?" not in asked:
            asked = platform_clarifying_question(missing_platforms)
        understood = parsed.get("understood_so_far")
        if not isinstance(understood, str) or not understood.strip():
            understood = (
                "Named accounts to scrape. Niche can come from the catalog; "
                "platform is still needed for "
                + ", ".join(f"@{h}" for h in missing_platforms)
                + "."
            )
        missing_fields = parsed.get("missing_fields")
        if not isinstance(missing_fields, list) or "platform" not in missing_fields:
            missing_fields = ["platform"]
        return AgentResult(
            ok=True,
            plan=None,
            clarifying_question=asked.strip(),
            understood_so_far=understood.strip(),
            missing_fields=missing_fields,
            latency_ms=elapsed,
            llm_latency_ms=llm_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    valid_codes = ctx.allowed_iso_codes
    # A region the operator names will contain countries our table does not
    # carry. Drop those and plan the rest, rather than failing the whole
    # request over one code nobody asked for by name.
    parsed, _ = filter_unsupported_countries(parsed, valid_codes)
    # Bring any out-of-range number into range before validating, and tell the
    # operator in the assumptions. Asking them to retype a whole request over
    # "5000 posts" is a worse answer than capping it and saying so.
    parsed, clamped = clamp_parameters(parsed)
    if clamped:
        existing = parsed.get("assumptions")
        parsed["assumptions"] = (
            list(existing) if isinstance(existing, list) else []
        ) + clamped
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
    if plan is None:
        return AgentResult(
            ok=False,
            plan=None,
            validation=validation,
            raw=raw_text,
            error="Validation failed: plan could not be parsed",
            error_code="validation_failed",
            latency_ms=elapsed,
            llm_latency_ms=llm_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
    return AgentResult(
        ok=True,
        plan=plan,
        flow=classify_flow(plan),
        latency_ms=elapsed,
        llm_latency_ms=llm_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
