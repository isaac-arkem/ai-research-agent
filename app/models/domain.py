# Domain models — the core business objects of the research agent.
#
# These define WHAT the AI must produce. Every field has constraints
# (e.g. max_creators can only be 5, 10, 20, 50, 100, or 200). Pydantic
# enforces these automatically — if the AI returns "max_creators: 75",
# validation fails before the operator ever sees it.

from __future__ import annotations

import re
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


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
    """One account-scrape job.

    Maps to the 'reference_profiles' pipeline. Posts-per-account, lookback,
    and niche are job-wide. Handles and platforms are listed together on this
    one object — never split into extra entries. max_creators does not apply.
    """

    pipeline: Literal["reference_profiles"]
    handles: List[str] = Field(min_length=1)
    platforms: List[Literal["tiktok", "instagram"]] = Field(min_length=1)
    niche: str
    posts_per_source: int = Field(default=10, ge=1, le=200)
    recency_days: Optional[int] = None
    title: Optional[str] = None
    rationale: str = Field(min_length=1)
    handle_platforms: Optional[Dict[str, Literal["tiktok", "instagram"]]] = None

    @model_validator(mode="before")
    @classmethod
    def coerce_legacy_shape(cls, data):
        if not isinstance(data, dict):
            return data
        handles = data.get("handles")
        if not handles and data.get("handle"):
            handles = [data["handle"]]
        if isinstance(handles, str):
            handles = [handles]
        cleaned = []
        if isinstance(handles, list):
            for item in handles:
                h = str(item).strip().lstrip("@")
                if h:
                    cleaned.append(h)
        platforms = data.get("platforms")
        if not platforms and data.get("platform"):
            platforms = [data["platform"]]
        if isinstance(platforms, str):
            platforms = [platforms]
        plat_list = []
        seen = set()
        if isinstance(platforms, list):
            for item in platforms:
                p = str(item).strip().lower()
                if p in ("tiktok", "instagram") and p not in seen:
                    seen.add(p)
                    plat_list.append(p)
        pairing = data.get("handle_platforms")
        if not isinstance(pairing, dict):
            pairing = {}
        cleaned_pair = {}
        for key, value in pairing.items():
            handle = str(key).strip().lstrip("@")
            if isinstance(value, list) and value:
                value = value[0]
            plat = str(value).strip().lower()
            if handle and plat in ("tiktok", "instagram"):
                cleaned_pair[handle] = plat
        if len(plat_list) == 1:
            for h in cleaned:
                cleaned_pair.setdefault(h, plat_list[0])
        recency = data.get("recency_days")
        if isinstance(recency, str) and recency.strip().lower() in ("any", "null", "none"):
            data = {**data, "recency_days": None}
        data = {**data, "handles": cleaned, "platforms": plat_list}
        if cleaned_pair:
            data["handle_platforms"] = cleaned_pair
        else:
            data.pop("handle_platforms", None)
        return data

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

    @model_validator(mode="before")
    @classmethod
    def coerce_string_lists(cls, data):
        if not isinstance(data, dict):
            return data
        for field in ("assumptions", "patterns_to_watch", "content_angles", "risks"):
            items = data.get(field)
            if not isinstance(items, list):
                continue
            cleaned = []
            for item in items:
                if isinstance(item, str):
                    if item.strip():
                        cleaned.append(item.strip())
                elif isinstance(item, dict):
                    text = (
                        item.get("text")
                        or item.get("value")
                        or item.get("note")
                        or "; ".join(str(v) for v in item.values() if v)
                    )
                    if text:
                        cleaned.append(str(text))
                elif item is not None:
                    cleaned.append(str(item))
            data[field] = cleaned
        return data

    @model_validator(mode="after")
    def collapse_into_one_job(self):
        """Account scrapes are one job: merge every row onto the first."""
        if len(self.reference_accounts) <= 1:
            return self
        first = self.reference_accounts[0]
        handles: List[str] = []
        seen_h = set()
        platforms: List[str] = []
        seen_p = set()
        pairing: Dict[str, Literal["tiktok", "instagram"]] = dict(
            first.handle_platforms or {}
        )
        niche = first.niche
        for account in self.reference_accounts:
            if not niche and account.niche:
                niche = account.niche
            default_plat = account.platforms[0] if len(account.platforms) == 1 else None
            for handle in account.handles:
                key = handle.lower()
                if key not in seen_h:
                    seen_h.add(key)
                    handles.append(handle)
                plat = (account.handle_platforms or {}).get(handle) or default_plat
                if plat:
                    pairing[handle] = plat
            for plat in account.platforms:
                if plat not in seen_p:
                    seen_p.add(plat)
                    platforms.append(plat)
        self.reference_accounts = [
            first.model_copy(
                update={
                    "handles": handles,
                    "platforms": platforms,
                    "niche": niche,
                    "handle_platforms": pairing or None,
                }
            )
        ]
        return self


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
    """One row of `apify_supported_countries`.

    Everything here is DB-owned. Add a country in Supabase and it appears in
    the agent's next context build — no code change, no redeploy.

    No apify_region_code field: every row in that table comes FROM the actor's
    supported list, so the ISO code IS the geo code. No seed_hashtags either —
    those live on `markets` and are not read here."""

    code: str
    iso: str
    name: str
    region: Optional[str] = None
    languages: List[str] = []
    # Optional nicknames from apify_supported_countries.aliases. The model
    # resolves the obvious ones ("Dubai", "KSA") unaided; this exists for
    # business shorthand it could not know, and to pin anything ambiguous.
    aliases: List[str] = []


class TaxonomyEntry(BaseModel):
    slug: str
    aliases: List[str]


class AgentContext(BaseModel):
    """Everything the agent needs to build its prompt: plannable countries
    and the topic taxonomy. Assembled once at startup.

    No country-alias table: the model resolves "Dubai", "KSA", "Türkiye" and
    "the Gulf" from ISO codes on its own — verified, and one less hardcoded
    list to drift out of date."""

    markets: List[MarketEntry]
    taxonomy: List[TaxonomyEntry]
    db_connected: bool = False

    @property
    def allowed_iso_codes(self) -> set:
        """Every country the agent may plan for — whatever is in `markets`."""
        return {m.iso for m in self.markets}

    @property
    def regions(self) -> Dict[str, List[str]]:
        """region -> ISO codes, built from markets.region.

        Replaces the hardcoded region map in the prompt. Re-file a market in
        Supabase and the expansion changes with it."""
        grouped: Dict[str, List[str]] = {}
        for market in self.markets:
            if not market.region:
                continue
            grouped.setdefault(market.region, [])
            if market.iso not in grouped[market.region]:
                grouped[market.region].append(market.iso)
        return grouped
