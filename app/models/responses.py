"""HTTP response bodies."""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from .domain import (
    Creator,
    FlowName,
    Hashtag,
    ComparisonBasis,
    MarketFinding,
    ResearchPlan,
    WebFinding,
)

_ASK_SUCCESS_EXAMPLE = {
    "ok": True,
    "conversation_id": "7f6117f9-f68f-4900-a653-0f676f771339",
    "message_id": "8e38d484-3179-4c29-a691-427315284cec",
    "flow": "discovery",
    "latency_ms": 5200,
    "llm_latency_ms": 4800,
    "prompt_tokens": 2100,
    "completion_tokens": 640,
    "plan": {
        "summary": "Find modest fashion creators in Saudi Arabia.",
        "assumptions": ["Platform defaulted to both TikTok and Instagram."],
        "recommended_runs": [
            {
                "pipeline": "creator_intelligence",
                "countries": ["SA"],
                "platforms": ["tiktok", "instagram"],
                "hashtags": ["modestfashion", "ازياء", "hijabstyle"],
                "niche": "fashion_beauty",
                "max_creators": 50,
                "posts_per_source": 25,
                "recency_days": None,
                "title": "SA modest fashion discovery",
                "rationale": "Hashtag discovery in the named market.",
            }
        ],
        "reference_accounts": [],
        "patterns_to_watch": ["Coverage of abaya vs western modest wear."],
        "content_angles": ["Day-to-night modest outfit breakdowns."],
        "risks": ["Arabic hashtag volume may be seasonal around Ramadan."],
    },
}


class AskResponse(BaseModel):
    """HTTP 200 from POST /ask. Errors use ErrorResponse, not this model."""

    model_config = ConfigDict(json_schema_extra={"example": _ASK_SUCCESS_EXAMPLE})

    ok: bool = True
    plan: Optional[ResearchPlan] = None
    clarifying_question: Optional[str] = None
    understood_so_far: Optional[str] = None
    missing_fields: Optional[List[str]] = None
    flow: Optional[FlowName] = None
    latency_ms: Optional[int] = None
    llm_latency_ms: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    conversation_id: Optional[str] = None
    message_id: Optional[str] = None
    # A review turn returns sources and no plan: the operator approves or
    # narrows them, and the plan is drawn on the turn after.
    findings: Optional[List[WebFinding]] = None
    creators: Optional[List[Creator]] = None
    hashtags: Optional[List[Hashtag]] = None
    # The countries a "which market" question resolved to. Without this on the
    # response the console has nothing to render and falls back to showing
    # only source cards — which reads as "go read them yourself".
    markets: Optional[List[MarketFinding]] = None
    # Ways to define "similar", for the operator to pick one. Absent unless
    # the question asked for similarity without saying what kind.
    comparison_bases: Optional[List[ComparisonBasis]] = None
    searched_for: Optional[str] = None
    awaiting_approval: Optional[bool] = None


class ErrorResponse(BaseModel):
    ok: bool = False
    error: str
    code: str
    details: Optional[object] = None
    conversation_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    markets: int
    taxonomy: int
    db_connected: bool
    auth_required: bool = False


class ConversationMessage(BaseModel):
    id: str
    role: str
    content: str
    flow: Optional[FlowName] = None
    plan: Optional[ResearchPlan] = None
    created_at: Optional[str] = None


class ConversationResponse(BaseModel):
    id: str
    title: Optional[str] = None
    messages: List[ConversationMessage] = Field(default_factory=list)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class ConversationSummary(BaseModel):
    id: str
    title: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class ConversationListResponse(BaseModel):
    conversations: List[ConversationSummary] = Field(default_factory=list)
