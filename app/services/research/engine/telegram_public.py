"""Telegram lane — public channels, read from their web preview.

Every public channel has a preview page at t.me/s/<channel> that anyone can
open without an account: the channel's name, subscriber count and its recent
posts with dates and views. Nothing here logs in or reads private chats.

Two steps:
  find     — which channels speak to the question. Tavily, limited to t.me,
             returns channel and post links; the channel names come out of
             those URLs.
  read     — each channel's preview page. When the search pointed at a
             particular post, the page is opened at that post.

The planner may use it on any question; the operator can also ask for it, or
leave it out, by name. Free apart from the one flat-rate Tavily search.
"""

from __future__ import annotations

import html
import logging
import re
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx

from app.services.search.base import SearchQuery
from app.services.search.tavily import TavilyProvider

logger = logging.getLogger(__name__)

PREVIEW = "https://t.me/s/{channel}"
DOMAIN = "t.me"
DEFAULT_MAX_CHANNELS = 5
# A channel page is ~20 posts, so five channels is ~100 — enough to push the
# web and news results out of the ranking. Keep the ones closest to the
# question, about as many as the news lane returns.
DEFAULT_MAX_POSTS = 20
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; researchAgent)"}

# A Telegram username: 5-32 characters, starts with a letter. This is the
# platform's URL grammar, not a list of opinions.
_USERNAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{3,31}$")
# Path segments of t.me that are features, not channels.
_NOT_CHANNELS = {
    "s", "joinchat", "addstickers", "addemoji", "addtheme", "share", "proxy",
    "socks", "iv", "c", "contact", "login", "setlanguage", "boost", "invoice",
}


def channels_in(urls: List[str]) -> List[Tuple[str, Optional[str]]]:
    """(channel, post id or None) for each channel the URLs point at, in order."""
    out: Dict[str, Optional[str]] = {}
    for url in urls:
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        if host not in ("t.me", "www.t.me", "telegram.me"):
            continue
        parts = [p for p in parsed.path.split("/") if p]
        if parts and parts[0] == "s":
            parts = parts[1:]
        if not parts or parts[0].lower() in _NOT_CHANNELS or parts[0].startswith("+"):
            continue
        name = parts[0]
        if not _USERNAME.match(name):
            continue
        post = parts[1] if len(parts) > 1 and parts[1].isdigit() else None
        key = name.lower()
        if key not in out or (out[key] is None and post):
            out[key] = post
    return list(out.items())


def _count(text: str) -> Optional[int]:
    """'1.56M' -> 1560000, '9.3K' -> 9300, '228' -> 228."""
    match = re.match(r"^\s*([\d.,]+)\s*([KkMm]?)", text or "")
    if not match:
        return None
    try:
        value = float(match.group(1).replace(",", ""))
    except ValueError:
        return None
    return int(value * {"k": 1_000, "m": 1_000_000}.get(match.group(2).lower(), 1))


def _plain(fragment: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", fragment, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text).strip()


def parse_preview(page: str, channel: str) -> List[dict]:
    """The posts on a channel's preview page, as lane items."""
    title_match = re.search(
        r'class="tgme_channel_info_header_title"[^>]*>(.*?)</div>', page, re.S)
    title = _plain(title_match.group(1)) if title_match else channel
    subs_match = re.search(
        r'<span class="counter_value">([^<]+)</span>\s*<span class="counter_type">subscribers',
        page)
    subscribers = _count(subs_match.group(1)) if subs_match else None

    items: List[dict] = []
    chunks = re.split(r'(?=<div class="tgme_widget_message_wrap)', page)
    for chunk in chunks:
        post = re.search(r'data-post="([^"]+)"', chunk)
        if not post:
            continue
        text_match = re.search(
            r'class="tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>', chunk, re.S)
        text = _plain(text_match.group(1)) if text_match else ""
        if not text:
            continue  # a photo or sticker with no words is nothing to read
        when = re.search(r'<time[^>]*datetime="([^"]+)"', chunk)
        views = re.search(r'class="tgme_widget_message_views">([^<]+)<', chunk)
        engagement = {}
        if views and _count(views.group(1)) is not None:
            engagement["views"] = _count(views.group(1))
        items.append({
            "id": f"TG-{post.group(1)}",
            "text": text,
            "url": f"https://t.me/{post.group(1)}",
            "handle": channel,
            "display_name": title,
            "date": when.group(1)[:10] if when else None,
            "engagement": engagement,
            "subscribers": subscribers,
            "relevance": 0.6,
            "why_relevant": "Telegram channel post",
        })
    return items


def read_channel(channel: str, post: Optional[str], timeout: float) -> List[dict]:
    url = PREVIEW.format(channel=channel) + (f"/{post}" if post else "")
    response = httpx.get(url, headers=HEADERS, timeout=timeout, follow_redirects=True)
    if response.status_code != 200:
        logger.info("telegram: %s answered HTTP %s", url, response.status_code)
        return []
    return parse_preview(response.text, channel)


def search_telegram(query: str, from_date: str, to_date: str, config: dict) -> List[dict]:
    """Posts from the public channels that speak to the query."""
    api_key = (config.get("TAVILY_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("no TAVILY_API_KEY to find channels with")
    timeout = float(config.get("TELEGRAM_TIMEOUT") or 15.0)
    provider = TavilyProvider(api_key, search_depth="basic", include_content=False)
    found = provider.search(SearchQuery(
        text=query,
        include_domains=[DOMAIN],
        limit=20,
        timeout=float(config.get("TAVILY_TIMEOUT") or 15.0),
    ))
    channels = channels_in([r.url for r in found])
    channels = channels[: int(config.get("TELEGRAM_MAX_CHANNELS") or DEFAULT_MAX_CHANNELS)]

    items: List[dict] = []
    for channel, post in channels:
        try:
            items.extend(read_channel(channel, post, timeout))
        except Exception as exc:  # one channel failing must not lose the rest
            logger.info("telegram: could not read %s: %s", channel, exc)
    read = len(items)
    items = closest_posts(items, query, int(config.get("TELEGRAM_MAX_POSTS") or DEFAULT_MAX_POSTS))
    logger.info("telegram: kept %d of %d post(s) from %d channel(s) for %r",
                len(items), read, len(channels), query[:80])
    return items


def closest_posts(items: List[dict], query: str, limit: int) -> List[dict]:
    """The posts that best match the query, by the engine's own relevance
    score — the one the ranker applies to every source — newest first on a tie."""
    from app.services.research.engine.relevance import token_overlap_relevance

    for item in items:
        item["relevance"] = round(token_overlap_relevance(query, item.get("text") or ""), 3)
    ranked = sorted(items, key=lambda i: (i["relevance"], i.get("date") or ""), reverse=True)
    return ranked[:max(1, limit)]
