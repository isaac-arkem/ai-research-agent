"""Research chat endpoints — question in, structured plan out."""

import json
import logging
from typing import Dict, Optional

from fastapi import APIRouter, Depends, Query, Request, Response

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
            cq_json = json.dumps(
                {
                    "clarifying_question": result.clarifying_question,
                    "understood_so_far": result.understood_so_far,
                    "missing_fields": result.missing_fields or [],
                },
                ensure_ascii=False,
            )
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
