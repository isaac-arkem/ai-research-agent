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
