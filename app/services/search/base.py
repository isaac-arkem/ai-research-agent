# Web search — one interface, one provider behind it.
#
# The agent needs current facts it cannot recall: which hashtags a niche is
# actually using in a market this week. Tavily answers that, returning page
# text along with each result.
#
# The interface stays even though there is only one implementation, and it is
# worth being clear why: it is what keeps that choice out of the callers. They
# ask for `results`. Should a second provider ever be worth adding, it is a
# class in this package, not an edit spreading through the agent.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol, runtime_checkable


class SearchError(Exception):
    """A search failed in a way the caller should know about.

    Deliberately distinct from "no results". A provider that returns an empty
    list is saying "the web has nothing"; raising this says "I could not ask".
    Conflating them is how a broken search hides for months.
    """


@dataclass
class SearchResult:
    url: str
    title: str
    description: str = ""
    # The page's text, as returned with the result. None when the provider
    # had none to give for that URL.
    content: Optional[str] = None


@dataclass
class SearchQuery:
    """What to search for, and the constraints that make it current and local.

    `country` and `window` are the whole point: an operator asking about Saudi
    fitness wants what is live in SA now, not a global evergreen listicle.
    """

    text: str
    # ISO-2, lowercase. None means no geo-targeting.
    country: Optional[str] = None
    # Full country name, e.g. "nigeria". Tavily geo-targets by NAME, not code.
    # Supplied by the caller from apify_supported_countries.name rather than a
    # hardcoded ISO->name table in here — that data already lives in the DB.
    country_name: Optional[str] = None
    # Recency: "d" past day, "w" past week, "m" past month, "y" past year.
    # None — the default — means no time filter, which is what you want unless
    # the operator asked for one. A window applied by default hides most of
    # the web to no purpose.
    window: Optional[str] = None
    limit: int = 5
    timeout: float = 15.0
    # Language hint for providers that accept one, e.g. "ar" for Saudi.
    language: Optional[str] = None


@runtime_checkable
class SearchProvider(Protocol):
    """What every provider must offer. Keep this small — anything a provider
    cannot honour becomes a silent difference in behaviour between them."""

    name: str

    def search(self, query: SearchQuery) -> List[SearchResult]:
        """Run one search.

        Returns [] when the search ran and found nothing. Raises SearchError
        when the search could not run — a bad key, a block, a changed page
        structure. Callers rely on that distinction to tell "no data" from
        "broken", so never swallow a failure into an empty list.
        """
        ...


@dataclass
class ProviderHealth:
    """Result of a canary check. A provider that returns zero for a query which
    should always work is broken, not empty."""

    provider: str
    ok: bool
    detail: str = ""
    results_seen: int = 0
    checks: List[str] = field(default_factory=list)
