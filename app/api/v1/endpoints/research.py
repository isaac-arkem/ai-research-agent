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


def _history_for_request(req: AskRequest, user_id: str):
    """Server store wins when conversation_id is set; otherwise client history."""
    if req.conversation_id:
        cid = str(req.conversation_id)
        conv = conversation_service.store.get(cid, user_id)
        if conv is None:
            raise_http(400, "Unknown conversation_id", "unknown_conversation")
        return cid, conversation_service.store.history(cid, user_id)
    cid = conversation_service.store.create(
        user_id, req.prompt.strip()[:80] or "Research"
    )
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
                stored["comparison_bases"] = [
                    b.model_dump() for b in (result.comparison_bases or [])
                ]
                stored["searched_for"] = result.searched_for
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
    req: AskRequest,
    ctx: AgentContext,
    settings: Settings,
    history,
) -> tuple[str, AgentResult]:
    model = req.model or settings.research_agent_model
    result = generate_research_plan(
        req.prompt,
        ctx,
        openai_key=settings.openai_api_key,
        model=model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        history=history,
        settings=settings,
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
    )


def _error_payload(result: AgentResult, cid: str) -> dict:
    details = None
    if result.validation and result.validation.errors:
        details = [e.model_dump() for e in result.validation.errors]
    return error_body(
        result.error or "Request failed",
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
    cid, history = _history_for_request(req, user.id)
    _persist_user_message(cid, user.id, req.prompt)
    model, result = _run_model(req, ctx, settings, history)
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

    cid, history = _history_for_request(req, user.id)
    _persist_user_message(cid, user.id, req.prompt)

    events: "queue.Queue" = queue.Queue()

    def on_progress(stage: str, **fields) -> None:
        events.put(("progress", {"stage": stage, **fields}))

    def run() -> None:
        try:
            model = req.model or settings.research_agent_model
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
                str(exc) or "Request failed", "planner_failed", conversation_id=cid,
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
