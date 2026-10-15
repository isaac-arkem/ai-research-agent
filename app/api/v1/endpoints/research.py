"""Research chat endpoints — question in, structured plan out."""

import json
import logging
from typing import Dict, Optional

import asyncio
import queue
import threading

from fastapi import APIRouter, Depends, Query, Request, Response
from fastapi.responses import StreamingResponse

from app.core.auth import AuthUser, get_current_user
from app.core.config import Settings, get_settings
from app.core.dependencies import get_agent_context
from app.core.errors import error_body, raise_http
from app.core.rate_limit import enforce_rate_limit
from app.models.domain import AgentContext, AgentResult, ChatTurn
from app.models.requests import AskRequest
from app.models.responses import (
    AskResponse,
    ConversationListResponse,
    ConversationResponse,
    ErrorResponse,
)
from app.services import conversations as conversation_service
from app.services.brief import (
    BriefError,
    brief_message,
    curate_upload,
    read_brief_text,
    track_message,
)
from app.services.agent import generate_research_plan
from app.services.grounding import render_findings_message
from app.services.audit import write_audit

logger = logging.getLogger(__name__)

router = APIRouter()

ERROR_HTTP = {
    "empty_prompt": 400,
    "openai_failed": 502,
    "json_extract_failed": 502,
    "validation_failed": 422,
}


def _rate_header_map(request: Request) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    remaining = getattr(request.state, "rate_limit_remaining", None)
    limit = getattr(request.state, "rate_limit_limit", None)
    if remaining is not None:
        headers["X-RateLimit-Remaining"] = str(remaining)
    if limit is not None:
        headers["X-RateLimit-Limit"] = str(limit)
    return headers


def _rate_headers(request: Request, response: Response) -> None:
    for key, value in _rate_header_map(request).items():
        response.headers[key] = value


def _read_brief(req: AskRequest) -> Optional[str]:
    """The uploaded brief's words, or None without one. A file that cannot be
    read is the operator's to fix: 400, before anything runs or is stored."""
    if req.brief is None:
        return None
    try:
        return read_brief_text(req.brief.filename, req.brief.content_base64)
    except BriefError as exc:
        raise_http(400, str(exc), exc.code)


def _user_message(req: AskRequest) -> str:
    """What the thread stores for the operator's turn. An approved track keeps
    its group, so the results that follow stay labelled with it."""
    if req.track:
        return track_message(req.track, req.prompt)
    return req.prompt


# A brief that could not be read into tracks: the model failed (upstream), or
# it read the brief and found nothing to research.
BRIEF_ERROR_HTTP = {"brief_failed": 502, "brief_no_tracks": 422}


def _brief_turn(
    req: AskRequest, brief_text: str, cid: str, user_id: str, settings: Settings,
) -> AskResponse:
    """Read the brief into research tracks for the operator to approve.

    Nothing is searched on this turn: an approved track comes back as an
    ordinary question and runs through the existing flow. Raises BriefError.
    """
    import time

    started = time.perf_counter()
    note = req.prompt if req.prompt.strip() else None
    _persist_user_message(cid, user_id, brief_message(req.brief.filename, note))
    brief = curate_upload(req.brief.filename, brief_text, note, settings=settings)
    message_id = None
    try:
        message_id = conversation_service.store.add_assistant_message(
            cid, user_id,
            json.dumps({"research_brief": brief.model_dump()}, ensure_ascii=False),
        )
    except Exception as exc:
        logger.warning("failed to persist research brief: %s", exc)
    latency_ms = int((time.perf_counter() - started) * 1000)
    write_audit(
        user_id=user_id,
        prompt=brief_message(req.brief.filename, note),
        model=getattr(settings, "grounding_escalation_model", None) or settings.research_agent_model,
        result=AgentResult(ok=True, latency_ms=latency_ms),
        status_code=200,
        conversation_id=cid,
        message_id=message_id,
    )
    return AskResponse(
        ok=True,
        conversation_id=cid,
        message_id=message_id,
        latency_ms=latency_ms,
        research_brief=brief,
        awaiting_approval=True,
    )


def _thread_title(req: AskRequest) -> str:
    if req.brief is not None:
        return f"Brief: {req.brief.filename}"[:80]
    return req.prompt.strip()[:80] or "Research"


def _history_for_request(req: AskRequest, user_id: str):
    """Server store wins when conversation_id is set; otherwise client history."""
    if req.conversation_id:
        cid = str(req.conversation_id)
        conv = conversation_service.store.get(cid, user_id)
        if conv is None:
            raise_http(400, "Unknown conversation_id", "unknown_conversation")
        return cid, conversation_service.store.history(cid, user_id)
    cid = conversation_service.store.create(user_id, _thread_title(req))
    history = [
        ChatTurn(role=m.role, content=m.content) for m in req.conversation_history
    ]
    return cid, history


def _persist_user_message(cid: str, user_id: str, prompt: str) -> None:
    try:
        conversation_service.store.add_user_message(cid, user_id, prompt)
    except Exception as exc:
        logger.warning("failed to persist user message: %s", exc)


def _persist_assistant(cid: str, user_id: str, result: AgentResult) -> Optional[str]:
    if result.ok and result.plan:
        try:
            return conversation_service.store.add_assistant_message(
                cid,
                user_id,
                json.dumps(result.plan.model_dump(), ensure_ascii=False),
                flow=result.flow,
                plan=result.plan,
            )
        except Exception as exc:
            logger.warning("failed to persist assistant message: %s", exc)
            return None
    if result.ok and result.clarifying_question:
        try:
            stored = {
                "clarifying_question": result.clarifying_question,
                "understood_so_far": result.understood_so_far,
                "missing_fields": result.missing_fields or [],
            }
            if result.findings:
                # The fenced copy, stored but never displayed. This is how the
                # sources reach the next turn: history hands this message back
                # to triage and to the planner, so the approval turn can plan
                # from what the operator actually approved.
                stored["web_results"] = render_findings_message(
                    result.findings,
                    result.searched_for or "",
                    creators=result.creators,
                    hashtags=result.hashtags,
                    markets=result.markets,
                )
                # The same sources structurally, for the console to render as
                # cards when a thread is reopened. The fenced copy above is
                # for the planner; this one is for the operator.
                stored["findings"] = [f.model_dump() for f in result.findings]
                stored["creators"] = [c.model_dump() for c in (result.creators or [])]
                stored["hashtags"] = [h.model_dump() for h in (result.hashtags or [])]
                stored["markets"] = [m.model_dump() for m in (result.markets or [])]
                stored["searched_for"] = result.searched_for
            if result.comparison_bases:
                # Outside the findings block on purpose: the turn that offers
                # these has NOT searched — asking what "similar" means is what
                # it does instead of searching. Stored in there, the options
                # vanished the moment the operator reopened the thread.
                stored["comparison_bases"] = [
                    b.model_dump() for b in result.comparison_bases
                ]
            if result.filters:
                # Read back by the next turn's filter reading.
                stored["filters"] = result.filters
            if result.comparison:
                # Read back so "more like him" keeps the basis already chosen.
                stored["comparison"] = result.comparison
            if result.concepts:
                # So a reopened thread shows them again.
                stored["concepts"] = [c.model_dump() for c in result.concepts]
            cq_json = json.dumps(stored, ensure_ascii=False)
            return conversation_service.store.add_assistant_message(
                cid,
                user_id,
                cq_json,
            )
        except Exception as exc:
            logger.warning("failed to persist clarifying question: %s", exc)
            return None
    return None


def _run_model(
    prompt: str,
    ctx: AgentContext,
    settings: Settings,
    history,
    concepts_for: Optional[str] = None,
) -> tuple[str, AgentResult]:
    model = settings.research_agent_model
    result = generate_research_plan(
        prompt,
        ctx,
        openai_key=settings.openai_api_key,
        model=model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        history=history,
        settings=settings,
        concepts_for=concepts_for,
    )
    return model, result


def _success_body(result: AgentResult, cid: str, message_id: Optional[str]) -> AskResponse:
    return AskResponse(
        ok=True,
        plan=result.plan,
        clarifying_question=result.clarifying_question,
        understood_so_far=result.understood_so_far,
        missing_fields=result.missing_fields,
        flow=result.flow,
        latency_ms=result.latency_ms,
        llm_latency_ms=result.llm_latency_ms,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        conversation_id=cid,
        message_id=message_id,
        findings=result.findings,
        creators=result.creators,
        hashtags=result.hashtags,
        markets=result.markets,
        comparison_bases=result.comparison_bases,
        searched_for=result.searched_for,
        awaiting_approval=result.awaiting_approval,
        concepts=result.concepts,
    )


def _error_payload(result: AgentResult, cid: str) -> dict:
    details = None
    if result.validation and result.validation.errors:
        details = [e.model_dump() for e in result.validation.errors]
    message = result.error or "Request failed"
    if result.error_code in {"openai_failed", "json_extract_failed", "planner_failed"}:
        message = "Upstream failed"
    return error_body(
        message,
        result.error_code or "upstream_error",
        details=details,
        conversation_id=cid,
    )


@router.post(
    "/ask",
    response_model=AskResponse,
    response_model_exclude_none=True,
    responses={
        400: {"model": ErrorResponse, "description": "Empty prompt or unknown conversation_id"},
        401: {"model": ErrorResponse, "description": "Missing or invalid Bearer token"},
        422: {"model": ErrorResponse, "description": "Request schema invalid, or plan failed validation"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        502: {"model": ErrorResponse, "description": "OpenAI or auth upstream failed"},
    },
)
def ask(
    req: AskRequest,
    request: Request,
    response: Response,
    settings: Settings = Depends(get_settings),
    ctx: AgentContext = Depends(get_agent_context),
    user: AuthUser = Depends(enforce_rate_limit),
):
    _rate_headers(request, response)
    brief_text = _read_brief(req)
    cid, history = _history_for_request(req, user.id)
    if brief_text is not None:
        try:
            return _brief_turn(req, brief_text, cid, user.id, settings)
        except BriefError as exc:
            raise_http(BRIEF_ERROR_HTTP.get(exc.code, 502), str(exc), exc.code,
                       conversation_id=cid)
    _persist_user_message(cid, user.id, _user_message(req))
    # A brief track's turn also builds concepts from its sources.
    model, result = _run_model(req.prompt, ctx, settings, history, req.track)
    status = 200 if result.ok else ERROR_HTTP.get(result.error_code or "", 502)
    message_id = _persist_assistant(cid, user.id, result)
    write_audit(
        user_id=user.id,
        prompt=req.prompt,
        model=model,
        result=result,
        status_code=status,
        conversation_id=cid,
        message_id=message_id,
    )
    if not result.ok:
        payload = _error_payload(result, cid)
        raise_http(
            status,
            payload["error"],
            payload["code"],
            details=payload.get("details"),
            conversation_id=cid,
        )
    return _success_body(result, cid, message_id)



# ── streaming ────────────────────────────────────────────────────────
#
# A review turn does three slow things in a row — triage, a 20-result
# advanced search, then reading every page — and the console had nothing to
# show for any of it. Same pipeline, same result; it just says what it is
# doing while it does it.
#
# Not token streaming: every model call here asks for a JSON object, so
# streaming the tokens would spell out `{"crea` `tors":` to nobody's benefit.
# The stages are the part a person wants to see.

STREAM_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    # Nginx buffers text/event-stream by default, which holds every event
    # until the response ends — exactly the silence this is here to fix.
    "X-Accel-Buffering": "no",
}

# The queue is unbounded but the producer is one turn of one request, so it
# holds a handful of events at most.
_DONE = object()


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/ask/stream", include_in_schema=True)
async def ask_stream(
    req: AskRequest,
    request: Request,
    user: AuthUser = Depends(enforce_rate_limit),
    ctx: AgentContext = Depends(get_agent_context),
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    """The same turn as POST /ask, narrated as it happens.

    The pipeline is synchronous — the OpenAI and Tavily clients both block —
    so it runs on a worker thread and posts its stages to a queue that this
    generator drains. The final event carries exactly the body /ask would
    have returned, so a client can ignore the rest and still be correct.
    """

    brief_text = _read_brief(req)
    cid, history = _history_for_request(req, user.id)
    if brief_text is None:
        _persist_user_message(cid, user.id, _user_message(req))

    events: "queue.Queue" = queue.Queue()

    def on_progress(stage: str, **fields) -> None:
        events.put(("progress", {"stage": stage, **fields}))

    def run() -> None:
        try:
            if brief_text is not None:
                on_progress("brief", detail="Reading the brief into research tracks")
                try:
                    body = _brief_turn(req, brief_text, cid, user.id, settings)
                except BriefError as exc:
                    events.put(("failed", error_body(str(exc), exc.code, conversation_id=cid)))
                    return
                events.put(("done", body.model_dump(exclude_none=True)))
                return
            model = settings.research_agent_model
            result = generate_research_plan(
                req.prompt,
                ctx,
                openai_key=settings.openai_api_key,
                model=model,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                history=history,
                settings=settings,
                on_progress=on_progress,
                concepts_for=req.track,
            )
            message_id = _persist_assistant(cid, user.id, result)
            status = 200 if result.ok else ERROR_HTTP.get(result.error_code or "", 500)
            write_audit(
                user_id=user.id, prompt=req.prompt, model=model, result=result,
                status_code=status, conversation_id=cid, message_id=message_id,
            )
            if result.ok:
                body = _success_body(result, cid, message_id).model_dump(
                    exclude_none=True
                )
                events.put(("done", body))
            else:
                events.put(("failed", _error_payload(result, cid)))
        except Exception as exc:  # noqa: BLE001 - the stream must always close
            logger.exception("streamed ask failed")
            events.put(("failed", error_body(
                "Upstream failed", "planner_failed", conversation_id=cid,
            )))
        finally:
            events.put((_DONE, None))

    threading.Thread(target=run, daemon=True).start()

    async def stream():
        # Sent immediately so the console can swap its spinner for a status
        # line before the first slow call has even started.
        yield _sse("progress", {"stage": "started", "conversation_id": cid,
                                "detail": "Working on it"})
        loop = asyncio.get_running_loop()
        while True:
            kind, payload = await loop.run_in_executor(None, events.get)
            if kind is _DONE:
                break
            if await request.is_disconnected():
                # Nobody is listening. The worker finishes and persists
                # anyway, so the turn is not lost from the conversation.
                break
            yield _sse(kind, payload)

    return StreamingResponse(
        stream(), media_type="text/event-stream", headers=STREAM_HEADERS
    )

@router.get(
    "/conversations",
    response_model=ConversationListResponse,
    response_model_exclude_none=True,
)
def list_conversations(
    user: AuthUser = Depends(get_current_user),
    limit: int = Query(50, ge=1, le=100),
):
    return ConversationListResponse(
        conversations=conversation_service.store.list(user.id, limit=limit)
    )


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationResponse,
    response_model_exclude_none=True,
)
def get_conversation(
    conversation_id: str,
    user: AuthUser = Depends(get_current_user),
):
    conv = conversation_service.store.get(conversation_id, user.id)
    if conv is None:
        raise_http(400, "Unknown conversation_id", "unknown_conversation")
    return conv
