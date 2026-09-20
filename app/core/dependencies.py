import logging
import re
from typing import List, Optional

from app.core.config import Settings, get_settings
from app.core.errors import raise_http
from app.core.supabase import get_supabase_admin
from app.models.domain import AgentContext, MarketEntry, TaxonomyEntry

logger = logging.getLogger(__name__)


class CountriesUnavailable(RuntimeError):
    """The country list could not be read and there is no fallback.

    Surfaced as 503 rather than 500: the service is fine, its reference data
    is not, and retrying once Supabase is reachable will work."""


COUNTRIES_TABLE = "apify_supported_countries"


def _fetch_countries_from_db(settings: Settings) -> Optional[List[MarketEntry]]:
    """Load plannable countries from `apify_supported_countries`.

    This table belongs to the research agent alone — `markets` stays the
    source of truth for the scrape pipelines and anything keyed on
    market_id. Returns None on any failure so the caller can raise
    CountriesUnavailable rather than plan against a stale list.
    """
    client = get_supabase_admin(settings)
    if client is None:
        return None

    try:
        rows = (
            client.table(COUNTRIES_TABLE)
            .select("country_code, name, region, aliases, languages")
            .eq("is_active", True)
            .execute()
            .data
        )

        countries = []
        seen = set()
        for row in rows:
            code = row.get("country_code")
            if not code or code in seen:
                continue
            seen.add(code)
            countries.append(
                MarketEntry(
                    code=code,
                    iso=code,
                    name=row.get("name") or code,
                    region=row.get("region"),
                    languages=row.get("languages") or [],
                    aliases=row.get("aliases") or [],
                )
            )

        if not countries:
            logger.warning(
                "%s returned 0 rows — no countries to plan against",
                COUNTRIES_TABLE,
            )
            return None

        logger.info("Loaded %d countries from %s", len(countries), COUNTRIES_TABLE)
        return countries

    except Exception as exc:
        logger.warning(
            "%s fetch failed (%s) — no countries to plan against",
            COUNTRIES_TABLE,
            exc,
        )
        return None


NICHES_TABLE = "niches"
REFERENCE_ACCOUNTS_TABLE = "reference_accounts"

# reference_accounts.topic is usage data, not a curated list — it carries
# smoke-test rows (khaby_test, codex_smoke, sona_test). Offering those to the
# model as real niches would put "codex_smoke" in a scrape plan. This is a
# hygiene rule on shape, not a hardcoded list of niches.
_TEST_SLUG_MARKERS = ("_test", "test_", "_smoke", "demo_")


# The plan schema requires niche to match ^[a-z0-9_]+$. Offering the model a
# slug it cannot legally emit invites a failed plan — the live data has "MAGA"
# alongside "maga", and reusing the former would fail validation.
_VALID_SLUG_RE = re.compile(r"^[a-z0-9_]+$")


def _looks_like_test_data(slug: str) -> bool:
    lowered = slug.lower()
    return any(marker in lowered for marker in _TEST_SLUG_MARKERS)


def _normalise_slug(raw: Optional[str]) -> Optional[str]:
    """Lowercase, and reject anything the plan schema would refuse."""
    slug = (raw or "").strip().lower()
    if not slug or _looks_like_test_data(slug) or not _VALID_SLUG_RE.match(slug):
        return None
    return slug


def _fetch_taxonomy_from_db(settings: Settings) -> List[TaxonomyEntry]:
    """Existing niche labels, merged from two places.

    `niches` is the curated table and the one new niches are created in — it
    is becoming the main source. `reference_accounts.topic` is what is
    actually in use, and still carries older slugs (music_dance, cooking_mum)
    that predate the table. Merging both means the agent recognises a niche
    whichever way it got into the system.

    This list is a HINT, not a constraint: the prompt tells the model to reuse
    a slug when one fits and coin a new one when none does. So unlike the
    country list, a failure here returns empty rather than raising — the agent
    still plans, it just might coin a slug that already exists.
    """
    client = get_supabase_admin(settings)
    if client is None:
        return []

    entries: dict = {}

    try:
        rows = (
            client.table(NICHES_TABLE)
            .select("slug, label, description, active")
            .eq("active", True)
            .execute()
            .data
        )
        for row in rows:
            slug = _normalise_slug(row.get("slug"))
            if not slug:
                continue
            # label and description are matching hints — they help the model
            # tie "cosplay in Dubai" to catgirls_dubai.
            hints = [h for h in (row.get("label"), row.get("description")) if h]
            entries[slug] = TaxonomyEntry(slug=slug, aliases=hints)
    except Exception as exc:
        logger.warning("%s fetch failed: %s", NICHES_TABLE, exc)

    try:
        rows = (
            client.table(REFERENCE_ACCOUNTS_TABLE)
            .select("topic")
            .execute()
            .data
        )
        for row in rows:
            slug = _normalise_slug(row.get("topic"))
            if not slug or slug in entries:
                continue
            entries[slug] = TaxonomyEntry(slug=slug, aliases=[])
    except Exception as exc:
        logger.warning("%s topic fetch failed: %s", REFERENCE_ACCOUNTS_TABLE, exc)

    if not entries:
        logger.warning("no niches loaded — the model will coin its own slugs")
    else:
        logger.info("Loaded %d existing niches", len(entries))
    return [entries[slug] for slug in sorted(entries)]


def build_context(settings: Optional[Settings] = None) -> AgentContext:
    """Assemble the agent's context: plannable countries from
    apify_supported_countries, plus existing niches from `niches` and
    reference_accounts.topic. Nothing is hardcoded.

    Called once at startup — the context doesn't change between requests."""

    if settings is None:
        settings = get_settings()

    countries = _fetch_countries_from_db(settings)
    if countries is None:
        # No hardcoded fallback on purpose. Planning against a stale copy of
        # the country list is worse than not planning: the operator gets a
        # confident plan for markets that may no longer be scrapeable, and
        # nothing in the response says the data was out of date.
        raise CountriesUnavailable(
            f"Could not load {COUNTRIES_TABLE} from Supabase. The research "
            "agent has no country list to plan against — check the Supabase "
            "credentials and that the table exists "
            "(sql/002_apify_supported_countries.sql)."
        )

    return AgentContext(
        markets=countries,
        taxonomy=_fetch_taxonomy_from_db(settings),
        db_connected=True,
    )


# Built lazily on first request so imports (and tests) don't hit the DB.
_agent_context: Optional[AgentContext] = None


def get_agent_context() -> AgentContext:
    """Dependency: provides the agent's context data to any route that needs it.

    Not cached on failure — a transient Supabase outage should not pin the
    process into a broken state until it is restarted.
    """
    global _agent_context
    if _agent_context is None:
        try:
            _agent_context = build_context()
        except CountriesUnavailable as exc:
            logger.error("agent context unavailable: %s", exc)
            raise_http(503, str(exc), "countries_unavailable")
    return _agent_context


def reset_agent_context() -> None:
    global _agent_context
    _agent_context = None
