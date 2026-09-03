# HTTP error contract for the research API.
#
# Status codes the client can rely on:
#   400  bad input that passed JSON schema but cannot be processed
#   401  missing or invalid Bearer token (when AUTH_REQUIRED=true)
#   422  request schema invalid, or the LLM plan failed validation
#   429  rate limit
#   502  upstream failure (OpenAI, or auth provider when checking a token)

from typing import Any, Optional

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


def error_body(
    message: str,
    code: str,
    *,
    details: Any = None,
    conversation_id: Optional[str] = None,
) -> dict:
    body = {"ok": False, "error": message, "code": code}
    if details is not None:
        body["details"] = details
    if conversation_id is not None:
        body["conversation_id"] = conversation_id
    return body


def raise_http(
    status: int,
    message: str,
    code: str,
    *,
    details: Any = None,
    conversation_id: Optional[str] = None,
    headers: Optional[dict] = None,
) -> None:
    raise HTTPException(
        status_code=status,
        detail=error_body(message, code, details=details, conversation_id=conversation_id),
        headers=headers,
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail
    if isinstance(detail, dict) and "ok" in detail:
        content = detail
    else:
        content = error_body(str(detail), "http_error")
    return JSONResponse(status_code=exc.status_code, content=content, headers=exc.headers)


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content=error_body(
            "Invalid request",
            "validation_error",
            details=exc.errors(),
        ),
    )
