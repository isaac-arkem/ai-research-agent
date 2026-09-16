# Config shim for the vendored last30days engine.
#
# Upstream's env.py is ~1,700 lines: an .env parser, a macOS Keychain reader, a
# pass(1) reader, a setup wizard's state machine, and lazy imports of browser
# cookie extraction and four X clients. The engine's source modules need almost
# none of it — across the 27 files we vendored, exactly four names are read:
#
#   KEYCHAIN_KEYS    http.py, to redact secret VALUES out of recorded fixtures
#   AUTH_STATUS_OK   providers.py, comparing config["OPENAI_AUTH_STATUS"]
#   get_x_source     providers.py, and X is not vendored
#   get_config       referenced in docstrings only
#
# So this replaces it. Config comes from app.core.config.Settings, which is
# where researchAgent's credentials already live; nothing here reads a file, a
# keyring, or a browser profile.
#
# Keep this file's PUBLIC NAMES stable. The other 26 files are verbatim copies
# of upstream and should stay diffable against it, which means changes belong
# here rather than in them.

from __future__ import annotations

from typing import Any, Dict, Optional

# Upstream's AuthStatus literal. Only the "ok" arm is ever compared against.
AUTH_STATUS_OK = "ok"

# Names treated as credentials. http.config_secret_values() uses this to keep
# secret values out of recorded fixtures, so a name missing here is a value
# that could be written to disk in plaintext — add generously, it costs
# nothing. Mirrors upstream KEYCHAIN_KEYS minus the sources we did not vendor,
# plus researchAgent's own.
KEYCHAIN_KEYS = (
    "OPENAI_API_KEY",
    "TAVILY_API_KEY",
    "APIFY_TOKEN",
    "APIFY_API_TOKEN",
    "SCRAPECREATORS_API_KEY",
    "SUPABASE_SECRET_KEY",
    "NEXT_PUBLIC_SUPABASE_PUBLIC_KEY",
)


def get_config(settings: Any = None) -> Dict[str, Any]:
    """The engine's config dict, built from researchAgent's Settings.

    The engine passes this dict down to every source module and reads keys out
    of it by name, so the NAMES below are the contract — they are what the
    vendored modules look for, not what researchAgent happens to call them.
    """
    if settings is None:
        from app.core.config import get_settings

        settings = get_settings()

    apify = getattr(settings, "apify_token", None)
    config: Dict[str, Any] = {
        "OPENAI_API_KEY": getattr(settings, "openai_api_key", None),
        # Present and "ok" together are what providers.py checks before it
        # will build an OpenAI client.
        "OPENAI_AUTH_STATUS": (
            AUTH_STATUS_OK if getattr(settings, "openai_api_key", None) else None
        ),
        "TAVILY_API_KEY": getattr(settings, "search_api_key", None),
        # "news" keeps dates (and survives normalize's require_date gate);
        # "general" geo-targets and returns evergreen directory pages. See
        # grounding.tavily_search for why that is a real trade-off.
        # "general" is what researchAgent's own provider always used: it is
        # the only topic Tavily geo-targets on, and it returns the parsed page
        # body the creator extraction reads. It carries no dates, which is
        # fine now that normalize no longer requires one.
        "TAVILY_TOPIC": getattr(settings, "tavily_topic", None) or "general",
        # The same two settings researchAgent's own Tavily provider reads, so
        # the vendored backend and the fallback provider ask for the same
        # thing. Vendoring had silently dropped both.
        "TAVILY_MAX_RESULTS": getattr(settings, "search_results_per_query", 20),
        "TAVILY_SEARCH_DEPTH": getattr(settings, "search_depth", "advanced"),
        # researchAgent's Settings calls this apify_token; the engine reads
        # APIFY_API_TOKEN. Same secret, two names — translate here rather than
        # renaming either side.
        "APIFY_API_TOKEN": apify,
        # No ScrapeCreators account. Left explicitly absent so the TikTok and
        # Instagram lanes take their Apify branch (both are gated on
        # `apify_token and not token`) instead of calling an API we cannot
        # authenticate against.
        "SCRAPECREATORS_API_KEY": None,
        # Upstream's pipeline read this to force the Apify path even when a
        # ScrapeCreators key existed. We have no such key, so the branch is
        # already forced — this is kept so the flag behaves as documented if
        # one is ever added.
        "SOCIAL_PROVIDER": "apify" if apify else None,
    }
    return {k: v for k, v in config.items() if v is not None}


def keyless_web_allowed(config: Dict[str, Any]) -> bool:
    """Whether the web lane may fall back to its keyless floor.

    Upstream allows it only when the host has no native search of its own.
    researchAgent always has Tavily, and the floor is a DuckDuckGo HTML
    scrape that returns nothing from a datacenter IP anyway — so the honest
    answer here is always False. A dead lane that reports `unreachable` is
    better than one that pretends to have searched.
    """
    return False


def get_x_source(config: Dict[str, Any], local_only: bool = False) -> Optional[str]:
    """No X backend. The X lanes were deliberately not vendored.

    Upstream walks an auth chain here across bird/xai/xurl/xquik. Returning
    None is the same answer it gives when nothing is configured, and
    providers.resolve_runtime() already handles that arm.
    """
    return None
