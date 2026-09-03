# Domain models — the core business objects of the research agent.
#
# These define WHAT the AI must produce. Every field has constraints
# (e.g. max_creators can only be 5, 10, 20, 50, 100, or 200). Pydantic
# enforces these automatically — if the AI returns "max_creators: 75",
# validation fails before the operator ever sees it.

from __future__ import annotations

import re
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


FlowName = Literal["discovery", "deep_research", "reference", "mixed", "off_topic"]


# ── What the agent produces ──────────────────────────────────────────


class RecommendedRun(BaseModel):
    """One scrape job the agent is recommending.
    Maps to the 'creator_intelligence' pipeline — discovers creators by hashtag."""

    pipeline: Literal["creator_intelligence"]
    countries: List[str] = Field(min_length=1)
    platforms: List[Literal["tiktok", "instagram"]] = Field(min_length=1)
    hashtags: List[str] = Field(min_length=3)
    niche: str
    max_creators: Literal[5, 10, 20, 50, 100, 200]
    posts_per_source: int = Field(ge=1, le=200)
    recency_days: Optional[int] = None
    title: str = Field(min_length=1)
    rationale: str = Field(min_length=1)

    @field_validator("niche")
    @classmethod
    def niche_must_be_slug(cls, v: str) -> str:
        if not re.match(r"^[a-z0-9_]+$", v):
            raise ValueError("niche must be lowercase letters, numbers, and underscores only")
        return v

    @field_validator("recency_days")
    @classmethod
    def recency_must_be_positive(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 1:
            raise ValueError("recency_days must be a positive number or null")
        return v


class ReferenceAccount(BaseModel):
    """A specific account the agent is recommending we scrape directly.
    Maps to the 'reference_profiles' pipeline — scrapes known accounts."""

    pipeline: Literal["reference_profiles"]
    handle: str = Field(min_length=1)
    platform: Literal["tiktok", "instagram"]
    niche: str
    posts_per_source: int = Field(default=10, ge=1, le=200)
    recency_days: Optional[int] = None
    rationale: str = Field(min_length=1)

    @field_validator("niche")
    @classmethod
    def niche_must_be_slug(cls, v: str) -> str:
        if not re.match(r"^[a-z0-9_]+$", v):
            raise ValueError("niche must be lowercase letters, numbers, and underscores only")
        return v


class ResearchPlan(BaseModel):
    """The complete output from the agent — everything the operator needs
    to decide whether to run the scrapes."""

    summary: str = Field(min_length=1)
    assumptions: List[str]
    recommended_runs: List[RecommendedRun]
    reference_accounts: List[ReferenceAccount]
    patterns_to_watch: List[str]
    content_angles: List[str]
    risks: List[str]


# ── Internal tracking objects ────────────────────────────────────────


class ValidationError_(BaseModel):
    """One specific thing that was wrong with the AI's output."""

    rule: int
    field: str
    message: str


class ValidationResult(BaseModel):
    """The result of checking the AI's output against our 9 rules.
    If valid=True, plan contains the parsed output. If not, errors
    lists exactly what went wrong so we can debug."""

    valid: bool
    errors: List[ValidationError_]
    plan: Optional[ResearchPlan] = None


class ChatTurn(BaseModel):
    """One prior turn, used to give the LLM conversation context."""

    role: Literal["user", "assistant"]
    content: str


class AgentResult(BaseModel):
    """The final package returned to whoever called the agent.

    On success (ok=True): plan, flow, and latency fields.
    On failure (ok=False): error, error_code, optional validation/raw for audit.
    HTTP 200 uses AskResponse (no nested validation.plan). Failures use ErrorResponse."""

    ok: bool
    plan: Optional[ResearchPlan] = None
    clarifying_question: Optional[str] = None
    understood_so_far: Optional[str] = None
    missing_fields: Optional[List[str]] = None
    validation: Optional[ValidationResult] = None
    raw: Optional[str] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    flow: Optional[FlowName] = None
    latency_ms: Optional[int] = None
    llm_latency_ms: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

    class Config:
        # Drop None fields from the JSON response so successful responses
        # are clean: {"ok": true, "plan": {...}} — no null clutter.
        json_encoders = {}

    def model_dump(self, **kwargs):
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(**kwargs)


# ── Context the agent needs to do its job ────────────────────────────


class MarketEntry(BaseModel):
    code: str
    iso: str
    name: str


class TaxonomyEntry(BaseModel):
    slug: str
    aliases: List[str]


class AgentContext(BaseModel):
    """Everything the agent needs to build its prompt: the list of markets,
    country aliases, and topic taxonomy. Assembled once at startup."""

    markets: List[MarketEntry]
    country_aliases: Dict[str, List[str]]
    taxonomy: List[TaxonomyEntry]
    db_connected: bool = False
