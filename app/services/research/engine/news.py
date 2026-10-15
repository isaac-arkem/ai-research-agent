"""News lane — published reporting, dated, through Tavily's news topic.

The web lane asks Tavily's general topic, which geo-targets but carries no
dates. Reporting is the opposite case: a dated account from a named outlet is
the evidence, and where the outlet sits matters less than when it published.
So this lane asks the news topic, inside the run's window, with the operator's
own question rather than the planner's keyword rewrite — Tavily does its own
query understanding (see the web lane in orchestrator.py).

The planner may use it on any question; the operator can also ask for it, or
leave it out, by name. Flat-rate like the web lane.
"""

from __future__ import annotations

import logging
from typing import List
from urllib.parse import urlparse

from app.services.search.base import SearchError, SearchQuery
from app.services.search.tavily import TavilyProvider

logger = logging.getLogger(__name__)


def _domain(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def search_news(query: str, from_date: str, to_date: str, config: dict) -> List[dict]:
    """Dated news articles for the query, published inside the window."""
    api_key = (config.get("TAVILY_API_KEY") or "").strip()
    if not api_key:
        raise SearchError("no TAVILY_API_KEY")
    provider = TavilyProvider(
        api_key, search_depth=str(config.get("TAVILY_SEARCH_DEPTH") or "advanced"),
    )
    results = provider.search(SearchQuery(
        text=query,
        topic="news",
        start_date=from_date,
        end_date=to_date,
        limit=int(config.get("TAVILY_MAX_RESULTS") or 20),
        timeout=float(config.get("TAVILY_TIMEOUT") or 15.0),
    ))
    items = []
    for i, r in enumerate(results, start=1):
        if not r.url or not r.title:
            continue
        items.append({
            "id": f"NW{i}",
            "title": r.title,
            "url": r.url,
            "source_domain": _domain(r.url),
            "snippet": r.description or "",
            "body": r.content or "",
            "date": r.published_date,
            "relevance": 0.85,
            "why_relevant": "News report",
        })
    logger.info("news: %d article(s) for %r", len(items), query[:80])
    return items
