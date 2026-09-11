#!/usr/bin/env python
"""Try Tavily by hand.

Drives the search on its own, without the agent around it — useful for
judging result quality for a niche and market, and for checking the key and
the response shape without spending a turn of the planner.

    python scripts/try_search.py "trending tiktok hashtags fitness" --country ng
    python scripts/try_search.py "modest fashion hashtags" --country sa --lang ar
    python scripts/try_search.py --health
    python scripts/try_search.py "fitness hashtags" --country ng --read

--read prints the first lines of the page text Tavily returned with each result,
which is what an LLM would actually be given. That is the honest test: the
titles usually look better than the pages do.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.services.search import (  # noqa: E402
    SearchError,
    SearchQuery,
    provider_from_settings,
)


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", help="what to search for")
    parser.add_argument("--country", default=None, help="ISO-2, e.g. ng")
    parser.add_argument(
        "--country-name", default=None,
        help="full name, e.g. nigeria — Tavily geo-targets by name, not code",
    )
    parser.add_argument("--lang", default=None, help="language hint, e.g. ar")
    parser.add_argument(
        "--window", default=settings.search_window,
        help="recency: d day, w week, m month, y year (default from .env)",
    )
    parser.add_argument("--limit", type=int, default=settings.search_results_per_query)
    parser.add_argument(
        "--read", action="store_true",
        help="show the page text an LLM would receive, not just its length",
    )
    parser.add_argument(
        "--repeat", type=int, default=1,
        help="run the query N times — shows the cache and throttle working",
    )
    parser.add_argument(
        "--health", action="store_true",
        help="canary check: is the provider working, or has the markup changed?",
    )
    args = parser.parse_args()

    try:
        provider = provider_from_settings(settings)
    except SearchError as exc:
        print(f"ERROR  {exc}")
        return 1

    if args.health:
        health = provider.health()
        print(f"provider  {health.provider}")
        print(f"ok        {health.ok}")
        print(f"results   {health.results_seen}")
        print(f"selectors {', '.join(health.checks) if health.checks else '-'}")
        print(f"detail    {health.detail}")
        return 0 if health.ok else 1

    if not args.query:
        parser.error("give a query, or use --health")

    query = SearchQuery(
        text=args.query,
        country=args.country,
        country_name=args.country_name,
        language=args.lang,
        window=args.window,
        limit=args.limit,
        timeout=settings.search_timeout,
    )
    print(f"provider  {provider.name}")
    print(f"query     {query.text!r}")
    print(f"country   {query.country or '(none)'}   window {query.window or '(none)'}"
          f"   lang {query.language or 'en'}\n")

    results = []
    for attempt in range(1, args.repeat + 1):
        t = time.perf_counter()
        try:
            results = provider.search(query)
        except SearchError as exc:
            # Deliberately NOT "no results" — the search could not run.
            print(f"SEARCH FAILED  {exc}")
            if "TAVILY_API_KEY" in str(exc):
                print("\nGet a free key at https://www.tavily.com (no card).")
                print("Then add TAVILY_API_KEY=... to .env")
            elif "429" in str(exc) or "circuit open" in str(exc):
                print("\nQuota or rate limit. The circuit breaker now backs off")
                print("automatically — check your usage at tavily.com.")
            else:
                print("\nRun --health to check the key and response shape.")
            return 1
        if args.repeat > 1:
            ms = (time.perf_counter() - t) * 1000
            st = provider.stats
            print(f"  run {attempt}: {len(results)} results in {ms:7.1f}ms   "
                  f"cache hits={st.hits} misses={st.misses} waits={st.throttled_waits}")
    if args.repeat > 1:
        print()

    if not results:
        print("no results — the search ran, the web had nothing")
        return 0

    for i, r in enumerate(results, 1):
        print(f"{i}. {r.title}")
        print(f"   {r.url}")
        if r.description:
            print(f"   {r.description[:150]}")
        if r.content and args.read:
            print(f"   [{len(r.content)} chars] {r.content[:220]}...")
        elif r.content:
            print(f"   [page content: {len(r.content)} chars]")
        elif args.read:
            print("   [no page content returned for this result]")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
