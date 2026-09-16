"""Web search retrieval via Brave Search, Exa, Serper, Parallel, or a keyless floor."""

from __future__ import annotations

import sys
import urllib.parse
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlparse

from . import http, log, schema
from app.services.search import SearchError, SearchQuery
from app.services.search.tavily import TavilyProvider


@dataclass(frozen=True)
class GroundedClaimText:
    """Candidate text with its exact primary evidence item."""

    candidate_id: str
    title: str
    summary: str
    item: schema.SourceItem


def claim_source_map(report: schema.Report) -> dict[str, GroundedClaimText]:
    """Expose only candidate claims that have a clean primary-item trace.

    Freshness verification deliberately starts here instead of scanning all
    report prose. A candidate without a primary ``SourceItem`` cannot produce
    an auditable per-claim verdict.
    """
    grounded: dict[str, GroundedClaimText] = {}
    for candidate in report.ranked_candidates:
        item = schema.candidate_primary_item(candidate)
        if item is None:
            continue
        grounded[candidate.candidate_id] = GroundedClaimText(
            candidate_id=candidate.candidate_id,
            title=candidate.title,
            summary=candidate.snippet or item.snippet or item.body,
            item=item,
        )
    return grounded


# ---------------------------------------------------------------------------
# Brave Search API
# ---------------------------------------------------------------------------

def _parse_serper_date(raw: str) -> str | None:
    if not raw:
        return None
    normalized = _normalize_date(raw)
    if normalized:
        return normalized
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw.strip(), fmt).date().isoformat()
        except ValueError:
            continue
    return None




# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def web_search(
    query: str,
    date_range: tuple[str, str],
    config: dict,
    backend: str = "auto",
) -> tuple[list[dict], dict]:
    """Web search — researchAgent's own Tavily provider, and nothing else.

    Upstream shipped five interchangeable backends here (Brave, Exa, Serper,
    Parallel, plus a keyless DuckDuckGo floor) because the skill must work on
    whatever key its host happens to have. researchAgent has one search
    provider, chosen deliberately, configured once, and already written:
    app/services/search/tavily.py.

    Re-implementing it inside the engine produced a second copy that drifted.
    It asked for 5 results where the settings said 20, hardcoded search_depth,
    and ran on topic="news" — where Tavily ignores the country parameter — so
    a Ghana query came back with World Cup coverage and the Korea Herald.

    So this calls the real provider. `backend` is accepted and ignored, for
    vendored callers that still pass it.

    Returns ([], artifact) on failure rather than raising, matching the other
    lanes: one dead lane must not end the run.
    """
    

    api_key = (config.get("TAVILY_API_KEY") or "").strip()
    if not api_key:
        return [], {"label": "tavily", "reason": "no TAVILY_API_KEY"}

    provider = TavilyProvider(
        api_key,
        search_depth=str(config.get("TAVILY_SEARCH_DEPTH") or "advanced"),
    )
    # Tavily geo-targets by country NAME, not ISO code, and only on the
    # general topic — which is the topic the provider sends.
    country_name = config.get("LAST30DAYS_COUNTRY") or None
    try:
        results = provider.search(SearchQuery(
            text=query,
            country_name=country_name,
            limit=int(config.get("TAVILY_MAX_RESULTS") or 20),
            timeout=float(config.get("TAVILY_TIMEOUT") or 30.0),
        ))
    except SearchError as exc:
        log.source_log("Web", f"Tavily unavailable: {exc}", tty_only=False)
        return [], {"label": "tavily", "reason": str(exc)}
    except Exception as exc:  # pragma: no cover - defensive
        log.source_log("Web", f"Tavily failed: {exc}", tty_only=False)
        return [], {"label": "tavily", "reason": f"{type(exc).__name__}: {exc}"}

    items = []
    for i, r in enumerate(results, start=1):
        if not r.url or not r.title:
            continue
        items.append({
            "id": f"TV{i}",
            "title": r.title,
            "url": r.url,
            "source_domain": _domain(r.url),
            "snippet": r.description or "",
            # The parsed page, which is what creator extraction reads.
            "body": r.content or "",
            # Tavily's general topic carries no dates. normalize no longer
            # requires one and the reranker weighs recency on its own.
            "date": None,
            "relevance": 0.85,
            "why_relevant": "Tavily web search",
        })

    artifact = {
        "label": "tavily",
        "webSearchQueries": [query],
        "resultCount": len(items),
        "country": country_name,
    }
    if items and not _reddit_excluded(config):
        # Best-effort secondary fetch on already-retrieved web results. HTTP
        # failures are isolated so a reddit.com 403 is not attributed to the
        # web source itself.
        with http.capture_failures():
            items = _enrich_reddit_items(items)
    return items, artifact


def _reddit_excluded(config: dict) -> bool:
    """Return True when EXCLUDE_SOURCES contains 'reddit'.

    Respects the same suppression knob the pipeline uses for source gating,
    so a user who set EXCLUDE_SOURCES=reddit doesn't get Reddit content
    smuggled back in via web-search URLs.
    """
    raw = (config.get("EXCLUDE_SOURCES") or "").split(",")
    return any(s.strip().lower() == "reddit" for s in raw)


def _enrich_reddit_items(items: list[dict]) -> list[dict]:
    """Enrich web search results that are Reddit URLs with thread body and comments.

    Claude Code's WebFetch blocks reddit.com, so the model can't retrieve
    Reddit content from web search results. This fetches it via the public
    JSON API (reddit.com/.../.json) which bypasses that restriction.

    Callers should gate this with EXCLUDE_SOURCES=reddit handling (see
    `_reddit_excluded`) so a user who explicitly excluded Reddit doesn't
    get Reddit content via web-search URLs.
    """
    from . import reddit_enrich
    from .reddit_enrich import RedditRateLimitError

    for item in items:
        url = item.get("url", "")
        if "reddit.com" not in url or "/comments/" not in url:
            continue
        try:
            thread_data = reddit_enrich.fetch_thread_data(url, timeout=8)
            if not thread_data:
                continue
            parsed = reddit_enrich.parse_thread_data(thread_data)
            # selftext lives under parsed["submission"], not at the top level
            selftext = (parsed.get("submission") or {}).get("selftext", "")
            if selftext:
                item["snippet"] = selftext[:2000]
            comments = parsed.get("comments", [])
            top = reddit_enrich.get_top_comments(comments)
            if top:
                item["top_comments"] = [
                    {"score": c.get("score", 0), "excerpt": (c.get("body") or "")[:200]}
                    for c in top[:5]
                ]
            item["enriched_via"] = "reddit_json_api"
        except RedditRateLimitError as exc:
            # Stop iterating to avoid flooding more 429s
            sys.stderr.write(f"[Web] Reddit rate-limited, halting enrichment: {exc}\n")
            break
        except Exception as exc:
            sys.stderr.write(f"[Web] Reddit enrichment failed for {url}: {exc}\n")
    return items


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_date(value: object) -> str | None:
    if value is None:
        return None
    parsed = dates.parse_date(str(value).strip())
    if not parsed:
        return None
    return parsed.date().isoformat()


def _serper_date_param(iso_date: str) -> str:
    """Convert YYYY-MM-DD to MM/DD/YYYY for Serper tbs parameter."""
    parts = iso_date.split("-")
    return f"{parts[1]}/{parts[2]}/{parts[0]}"


def _in_date_range(pub_date: str | None, date_range: tuple[str, str]) -> bool:
    if not pub_date:
        return False
    return date_range[0] <= pub_date <= date_range[1]


def _domain(url: str) -> str:
    return urlparse(url).netloc.strip().lower()
