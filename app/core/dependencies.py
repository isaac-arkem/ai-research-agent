import logging
from typing import List, Optional

from app.core.config import Settings, get_settings
from app.core.supabase import get_supabase_admin
from app.models.domain import AgentContext, MarketEntry, TaxonomyEntry
from app.data.markets import CREATOR_MARKETS
from app.data.taxonomy import TAG_ALIASES
from app.data.aliases import COUNTRY_ALIASES

logger = logging.getLogger(__name__)


def _fetch_markets_from_db(settings: Settings) -> Optional[List[MarketEntry]]:
    """Try to load markets from Supabase. Returns None on any failure."""

    client = get_supabase_admin(settings)
    if client is None:
        return None

    try:
        result = (
            client
            .table("markets")
            .select("id, country_code, name, region, platform")
            .execute()
        )
        rows = result.data

        seen = set()
        markets = []
        for row in rows:
            code = row.get("country_code", "")
            if code and code not in seen:
                seen.add(code)
                markets.append(MarketEntry(
                    code=code,
                    iso=code,
                    name=row.get("name", code),
                ))

        if not markets:
            logger.warning("Supabase returned 0 markets — falling back to hardcoded")
            return None

        logger.info("Loaded %d markets from Supabase", len(markets))
        return markets

    except Exception as exc:
        logger.warning("Supabase fetch failed (%s) — falling back to hardcoded markets", exc)
        return None


def _build_hardcoded_markets() -> List[MarketEntry]:
    """The fallback: the 19 markets hardcoded in app/data/markets.py."""
    return [
        MarketEntry(code=m.code, iso=m.iso, name=m.name)
        for m in CREATOR_MARKETS
    ]


def _build_taxonomy() -> List[TaxonomyEntry]:
    """Build the taxonomy from the hardcoded aliases.
    The aliases tell the AI how to map natural language to niche slugs
    (e.g. "cooking" → cooking_mum). This is prompt-engineering data,
    not raw DB data, so it stays hardcoded until we create a
    niche_aliases table in Supabase."""
    return [
        TaxonomyEntry(slug=slug, aliases=aliases)
        for slug, aliases in TAG_ALIASES.items()
    ]


def build_context(settings: Optional[Settings] = None) -> AgentContext:
    """Assemble the agent's context: markets from DB (or fallback) +
    taxonomy aliases (always hardcoded).

    Called once at startup — the context doesn't change between requests."""

    if settings is None:
        settings = get_settings()

    db_markets = _fetch_markets_from_db(settings)
    markets = db_markets if db_markets is not None else _build_hardcoded_markets()

    return AgentContext(
        markets=markets,
        country_aliases=COUNTRY_ALIASES,
        taxonomy=_build_taxonomy(),
        db_connected=db_markets is not None,
    )


# Built lazily on first request so imports (and tests) don't hit the DB.
_agent_context: Optional[AgentContext] = None


def get_agent_context() -> AgentContext:
    """Dependency: provides the agent's context data to any route that needs it."""
    global _agent_context
    if _agent_context is None:
        _agent_context = build_context()
    return _agent_context


def reset_agent_context() -> None:
    global _agent_context
    _agent_context = None


def get_openai_key() -> str:
    """Dependency: provides the OpenAI API key from settings."""
    return get_settings().openai_api_key


def get_model() -> str:
    """Dependency: provides the configured LLM model name."""
    return get_settings().research_agent_model
