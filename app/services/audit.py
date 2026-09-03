# Audit log — latency, prompt tracking, error capture for every /ask.

import hashlib
import logging
from typing import Optional

from app.core.supabase import get_supabase_admin
from app.models.domain import AgentResult

logger = logging.getLogger("app.audit")

AUDIT_TABLE = "research_audit_logs"


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def write_audit(
    *,
    user_id: str,
    prompt: str,
    model: str,
    result: AgentResult,
    status_code: int,
    conversation_id: Optional[str] = None,
    message_id: Optional[str] = None,
) -> None:
    preview = prompt[:240]
    hashed = prompt_hash(prompt)
    logger.info(
        "ask user=%s conv=%s flow=%s ok=%s status=%s latency_ms=%s llm_ms=%s "
        "prompt_hash=%s prompt_tokens=%s completion_tokens=%s error_code=%s",
        user_id,
        conversation_id,
        result.flow,
        result.ok,
        status_code,
        result.latency_ms,
        result.llm_latency_ms,
        hashed,
        result.prompt_tokens,
        result.completion_tokens,
        result.error_code,
    )

    client = get_supabase_admin()
    if client is None:
        return
    try:
        client.table(AUDIT_TABLE).insert(
            {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "message_id": message_id,
                "prompt_hash": hashed,
                "prompt_preview": preview,
                "model": model,
                "flow": result.flow,
                "ok": result.ok,
                "error": result.error,
                "error_code": result.error_code,
                "latency_ms": result.latency_ms,
                "llm_latency_ms": result.llm_latency_ms,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "status_code": status_code,
            }
        ).execute()
    except Exception as exc:
        logger.warning("audit insert failed: %s", exc)
