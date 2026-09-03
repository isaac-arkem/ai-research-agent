"""Health check."""

from fastapi import APIRouter, Depends

from app.core.config import get_settings
from app.core.dependencies import get_agent_context
from app.models.domain import AgentContext
from app.models.responses import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check(ctx: AgentContext = Depends(get_agent_context)):
    return HealthResponse(
        status="ok",
        markets=len(ctx.markets),
        taxonomy=len(ctx.taxonomy),
        db_connected=ctx.db_connected,
        auth_required=get_settings().auth_required,
    )
