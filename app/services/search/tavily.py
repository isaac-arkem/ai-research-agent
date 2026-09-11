# Tavily provider — search built for agents rather than for people.
#
# It returns page CONTENT, not just links: the text is fetched and extracted
# server-side and comes back pre-chunked for a model, so nothing downstream
# has to retrieve or truncate pages.
#
# Two quirks worth knowing before reading the code:
#
#   * country is a NAME, not an ISO code — "nigeria", not "NG". The names live
#     in apify_supported_countries.name, so the CALLER supplies country_name
#     rather than this module carrying a hardcoded ISO->name table.
#   * search_depth decides both relevance and cost. advanced returns several
#     semantically relevant snippets per URL and costs 2 credits; basic, fast
#     and ultra-fast return one and cost 1. Default here is advanced: we are
#     asking narrow questions about a niche in a market, which is exactly the
#     case the extra credit buys something for.
#
# Docs: https://docs.tavily.com/documentation/api-reference/endpoint/search

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import httpx

from app.services.search.base import (
    ProviderHealth,
    SearchError,
    SearchQuery,
    SearchResult,
)

logger = logging.getLogger(__name__)

ENDPOINT = "https://api.tavily.com/search"

# Tavily accepts our window letters directly — d, w, m, y.
_VALID_WINDOWS = {"d", "w", "m", "y"}

# 0-20 per the API.
MAX_RESULTS = 20

# Statuses that mean "stop asking for a while". Worded so looks_like_a_block()
# in policy.py matches them and opens the circuit — otherwise a quota problem
# burns the rest of the month on calls that are already being refused.
# 432/433 are Tavily's plan/usage limits.
_QUOTA_STATUSES = {429, 432, 433}


# advanced costs 2 credits and returns several relevant snippets per URL.
# The other three cost 1 and return one.
DEPTHS = {"advanced", "basic", "fast", "ultra-fast"}


class TavilyProvider:
    """Search via Tavily. Needs TAVILY_API_KEY."""

    name = "tavily"

    def __init__(
        self,
        api_key: Optional[str],
        endpoint: str = ENDPOINT,
        *,
        search_depth: str = "advanced",
        include_content: bool = True,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.endpoint = endpoint
        # A typo here would come back as an opaque HTTP 400 from Tavily, so
        # it is caught at construction where the message can say what is wrong.
        if search_depth not in DEPTHS:
            raise SearchError(
                f"unknown tavily search_depth {search_depth!r}. "
                f"Known: {', '.join(sorted(DEPTHS))}"
            )
        self.search_depth = search_depth
        # Ask for the parsed page text alongside the snippet.
        self.include_content = include_content

    def _body(self, query: SearchQuery) -> Dict[str, object]:
        """Only send what the caller set — an empty country is not the same
        request as no country at all."""
        body: Dict[str, object] = {
            "query": query.text,
            "max_results": max(1, min(query.limit, MAX_RESULTS)),
            "search_depth": self.search_depth,
            "topic": "general",
        }
        # Geo-targeting is by name, and only honoured when topic is general.
        if query.country_name:
            body["country"] = query.country_name.strip().lower()
        if (query.window or "") in _VALID_WINDOWS:
            body["time_range"] = query.window
        if self.include_content:
            # markdown keeps structure; lists of hashtags survive better than
            # they do in flattened plain text.
            body["include_raw_content"] = "markdown"
        return body

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code == 200:
            return
        if response.status_code == 401:
            raise SearchError(
                "tavily rejected the API key (HTTP 401) — check TAVILY_API_KEY"
            )
        if response.status_code in _QUOTA_STATUSES:
            raise SearchError(
                f"tavily rate limit or credits exhausted (HTTP "
                f"{response.status_code})"
            )
        if response.status_code == 400:
            # Usually a malformed parameter — an unsupported country name is
            # the likely culprit, so say so.
            raise SearchError(
                f"tavily rejected the request (HTTP 400): "
                f"{response.text[:200]}"
            )
        raise SearchError(f"tavily returned HTTP {response.status_code}")

    def search(self, query: SearchQuery) -> List[SearchResult]:
        if not self.api_key:
            raise SearchError(
                "TAVILY_API_KEY is not set — get one free at "
                "https://www.tavily.com (no card required)"
            )

        try:
            response = httpx.post(
                self.endpoint,
                json=self._body(query),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=query.timeout,
            )
        except Exception as exc:
            raise SearchError(f"tavily request failed: {exc}") from exc

        self._raise_for_status(response)

        try:
            body = response.json()
        except Exception as exc:
            raise SearchError(f"tavily returned invalid JSON: {exc}") from exc

        credits = (body.get("usage") or {}).get("credits")
        if credits:
            logger.debug("tavily search used %s credit(s)", credits)

        return _collect_results(body, query.limit)


def _collect_results(body: dict, limit: int) -> List[SearchResult]:
    """Pull results out of the response.

    `content` is Tavily's relevance-selected snippets; `raw_content` is the
    full parsed page when include_raw_content was set. We keep the snippet as
    the description and the full text as content, so a caller can use either
    without retrieving the page separately.
    """
    rows = body.get("results") if isinstance(body, dict) else None
    if not isinstance(rows, list):
        return []

    out: List[SearchResult] = []
    for row in rows[:limit]:
        if not isinstance(row, dict):
            continue
        url = row.get("url") or ""
        title = row.get("title") or ""
        if not url or not title:
            continue
        out.append(
            SearchResult(
                url=url,
                title=title,
                description=row.get("content") or "",
                # None when include_raw_content was off — callers fall back
                content=row.get("raw_content") or None,
            )
        )
    return out


def health_check(provider: TavilyProvider, canary: str = "wikipedia") -> ProviderHealth:
    """Canary: a query that must return results.

    Catches an expired key, exhausted credits, or a changed response envelope
    before any of those quietly starve the agent of live data.
    """
    if not provider.api_key:
        return ProviderHealth(provider.name, False, "TAVILY_API_KEY is not set")
    try:
        results = provider.search(
            SearchQuery(text=canary, window=None, limit=5, country=None)
        )
    except SearchError as exc:
        return ProviderHealth(provider.name, False, str(exc))
    if not results:
        return ProviderHealth(
            provider.name,
            False,
            "no results for a canary query — the response shape may have changed",
        )
    with_content = sum(1 for r in results if r.content)
    return ProviderHealth(
        provider.name,
        True,
        "ok",
        len(results),
        [f"results={len(results)}", f"with_page_content={with_content}"],
    )


# Bound as a method so the provider satisfies the same shape as the others.
TavilyProvider.health = health_check  # type: ignore[attr-defined]
