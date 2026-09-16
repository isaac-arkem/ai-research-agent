# Configuration — all settings in one place.
#
# Instead of scattering os.getenv() calls across the codebase, we define
# every setting here. Pydantic's BaseSettings automatically reads from
# environment variables and validates them.
#
# Benefits:
#   - One place to see every setting the app needs
#   - Type-safe (if OPENAI_API_KEY is missing, the app fails at startup, not mid-request)
#   - Easy to override in tests
#
# How it reads values (in order):
#   1. Environment variables (e.g. OPENAI_API_KEY=sk-...)
#   2. A .env file in the project root
#   3. The default value defined here

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Every configuration value the app needs."""

    # Required — the app won't start without this
    openai_api_key: str

    # Supabase — the agent's country list comes from apify_supported_countries
    # and there is no hardcoded fallback, so requests 503 without it. Still
    # Optional[] so the process can boot and report the problem on /health
    # rather than crashing at import.
    # Same env var names as arkgpt: NEXT_PUBLIC_SUPABASE_URL + SUPABASE_SECRET_KEY
    next_public_supabase_url: Optional[str] = None
    supabase_secret_key: Optional[str] = None

    # Optional — same names as arkgpt so one .env works for both
    next_public_supabase_public_key: Optional[str] = None
    next_public_supabase_anon_key: Optional[str] = None

    # ── Web search ──────────────────────────────────────────────────────
    # Tavily, and only Tavily. See app/services/search/ for why: it returns
    # page content rather than links, so nothing downstream has to fetch and
    # lose half its pages to bot walls. There is no provider setting — the
    # choice is made, and a knob with one position only invites misconfiguring
    # production. Adding a second provider later is a code change, not an
    # env var.
    # https://www.tavily.com — 1,000 free credits a month.
    tavily_api_key: Optional[str] = None
    # advanced | basic | fast | ultra-fast. advanced returns several relevant
    # snippets per page instead of one, which is what the creator extraction
    # reads — so it costs 2 credits a search instead of 1 and is worth it.
    # Drop to basic to halve the cost if recall stops mattering.
    search_depth: str = "advanced"

    @property
    def search_api_key(self) -> Optional[str]:
        """The search provider's key. Named for the seam, not the vendor, so
        callers do not have to care which provider is behind it."""
        return self.tavily_api_key

    search_timeout: float = 15.0
    # Tavily's ceiling. Credits are charged per SEARCH, not per result, so
    # asking for fewer buys nothing — and the whole job is finding creators,
    # which more pages means more of. Relevance does fall off down the list;
    # the extractor drops the directories and agency pages that show up there.
    search_results_per_query: int = 20
    # Recency filter for the by-hand script only: d day, w week, m month,
    # y year, or empty for none. The agent does not read this — the window
    # comes from what the operator asked for, or is left off entirely.
    search_window: str = ""

    # ── Web grounding ───────────────────────────────────────────────────
    # Search the web before planning, so hashtags and creators come from this
    # week rather than the model's training data. See app/services/grounding.py.
    # Off switch, not a rewrite: with this false the planner behaves exactly as
    # it did before grounding existed.
    search_grounding_enabled: bool = True
    # Frames the search query and decides whether to search at all. A small
    # model is the right tool — it writes one line of JSON, and it runs on
    # every ask, so gpt-4o's price for it would be paid on every request.
    grounding_model: str = "gpt-4o-mini"

    # ── Multi-source research engine ────────────────────────────────────
    # With this on, a search fans out across Reddit, Hacker News, Instagram,
    # TikTok and Polymarket as well as the web, and the engine decides which
    # of them the question actually needs. Off, grounding calls the web
    # provider directly exactly as it did before — the same off-switch shape
    # as search_grounding_enabled, for the same reason: a bad day for the
    # engine must not become a bad day for the planner.
    research_engine_enabled: bool = True
    # Writes the query plan and resolves hashtags. NOT the grounding model:
    # gpt-4o-mini was measured emitting plans whose source_weights and
    # per-subquery sources contradicted each other, which routes a run to the
    # wrong lanes. This runs once per search rather than once per turn, so the
    # better model is affordable here.
    research_plan_model: str = "gpt-4o"
    # How far back a research run looks. Wide by default: a creator-landscape
    # question is not a news question, and a two-year-old "top creators" page
    # is often the best evidence there is. The reranker prefers recent items
    # on its own, so this does not have to.
    research_window_days: int = 365
    # quick | default | deep.
    #
    # NOT just a result-count knob. planner._sanitize_plan hard-truncates a
    # "quick" plan to ONE subquery:
    #
    #     if depth == "quick" and subqueries:
    #         subqueries = subqueries[:1]
    #
    # So quick cannot cover two platforms, two regions, or two angles — it
    # asks one question and reports whatever that one question found. It is
    # why a "which country" run only ever looked at Africa, and why a creator
    # question with no platform named came back all-TikTok: the Instagram
    # subquery was written and then dropped.
    #
    # "default" costs more — more subqueries, and DEPTH_LIMITS doubles
    # per-lane results from 10 to 20, which on the Apify lanes is real money.
    # Breadth is the thing that was wrong, so it is worth paying for.
    research_depth: str = "default"

    # Apify — used only to refresh the geo-targetable country list at startup.
    # Without it the captured list in app/data/apify_countries.py is used.
    apify_token: Optional[str] = None
    apify_tiktok_actor: str = "clockworks~tiktok-scraper"

    # Optional — sensible defaults
    research_agent_model: str = "gpt-4o"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 4000

    # Server settings
    app_name: str = "Research Agent"
    debug: bool = False

    # Auth: mirrors arkgpt requireSocialListeningUser (Bearer JWT).
    # Off by default so local chat works without a logged-in session.
    # Set AUTH_REQUIRED=true when this service is exposed next to arkgpt.
    auth_required: bool = False

    # Rate limit for POST /ask (per user id, or client IP when anonymous)
    rate_limit_requests: int = 20
    rate_limit_window_seconds: int = 60

    # CORS — comma-separated origins. Empty = allow localhost defaults.
    cors_origins: str = "http://localhost:3000,http://localhost:8000"

    @property
    def supabase_configured(self) -> bool:
        return bool(self.next_public_supabase_url and self.supabase_secret_key)

    @property
    def supabase_anon_key(self) -> Optional[str]:
        return self.next_public_supabase_public_key or self.next_public_supabase_anon_key

    @property
    def cors_origin_list(self) -> list:
        origins = [o.strip() for o in self.cors_origins.split(",") if o.strip()]
        return origins or ["http://localhost:3000", "http://localhost:8000"]

    class Config:
        # Tells Pydantic to also check a .env file for values
        env_file = ".env"
        env_file_encoding = "utf-8"
        # The .env file may contain vars for other services (arkgpt, Jenkins, etc.)
        # — ignore anything we don't need rather than crashing on it.
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Create the settings object once and cache it forever.
    lru_cache means the .env file is only read once, at first access."""
    return Settings()
