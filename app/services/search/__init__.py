# Web search — Tavily.
#
# The caller asks for results and gets them. Nothing in between: no cache, no
# throttle, no stored copy of what came back. A search costs one credit and
# runs every time it is asked for.
#
# Search results live in the conversation and nowhere else — the review turn
# writes them into the thread as a message, which is what the operator reads
# and what the planner reads back on the approval turn.
#
# The SearchProvider protocol in base.py is still the seam. Adding a second
# provider later means writing a class with .search(SearchQuery) ->
# [SearchResult] and choosing between them here; nothing that CALLS search
# has to change.

from __future__ import annotations

import logging
from typing import Optional

from app.services.search.base import (  # noqa: F401  (re-exported)
    ProviderHealth,
    SearchError,
    SearchProvider,
    SearchQuery,
    SearchResult,
)
from app.services.search.tavily import TavilyProvider

logger = logging.getLogger(__name__)


def build_provider(
    api_key: Optional[str] = None,
    *,
    search_depth: str = "advanced",
) -> SearchProvider:
    """The search provider."""
    return TavilyProvider(api_key, search_depth=search_depth)


def provider_from_settings(settings) -> SearchProvider:
    """The search provider, with the key and depth from config."""
    return build_provider(
        settings.search_api_key,
        search_depth=getattr(settings, "search_depth", "advanced"),
    )
