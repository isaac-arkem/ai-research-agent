"""Apify-based TikTok and Instagram search for /last30days.

Alternative provider to ScrapeCreators. Runs Apify Store actors
synchronously via the run-sync-get-dataset-items REST endpoint and maps
results to the same normalized item shape produced by tiktok.py and
instagram.py, so the rest of the pipeline stays provider-agnostic.

Requires APIFY_API_TOKEN in config (~/.config/last30days/.env).
Activated when SOCIAL_PROVIDER=apify is set, or automatically when no
SCRAPECREATORS_API_KEY is configured.

Actors used (pay-per-result, billed to the Apify account):
- clockworks/tiktok-scraper        (~$0.0023/result)
- apify/instagram-hashtag-scraper  (~$0.0021/result)
- apify/instagram-post-scraper     (~$0.0013/post, creator profiles only)
"""

from typing import Any, Dict, List, Optional

from . import dates, http, log
from .relevance import token_overlap_relevance as _compute_relevance

APIFY_BASE = "https://api.apify.com/v2"
TIKTOK_ACTOR = "clockworks~tiktok-scraper"
IG_HASHTAG_ACTOR = "apify~instagram-hashtag-scraper"
IG_POSTS_ACTOR = "apify~instagram-post-scraper"

# Results per query/hashtag by depth. Kept modest: actors are pay-per-result.
DEPTH_LIMITS = {"quick": 10, "default": 20, "deep": 40}


def _log(msg: str):
    log.source_log("Apify", msg)


def _run_actor(
    actor: str,
    input_data: Dict[str, Any],
    token: str,
    timeout: int = 240,
) -> List[Dict[str, Any]]:
    """Run an Apify actor synchronously and return its dataset items.

    retries=1 (single attempt) on purpose: a retried run is a second
    actor run and bills a second time.
    """
    url = f"{APIFY_BASE}/acts/{actor}/run-sync-get-dataset-items?format=json&clean=true"
    data = http.post(
        url,
        json_data=input_data,
        headers={"Authorization": f"Bearer {token}"},
        timeout=timeout,
        retries=1,
    )
    if isinstance(data, list):
        return [i for i in data if isinstance(i, dict)]
    if isinstance(data, dict):
        err = data.get("error")
        if err:
            raise http.HTTPError(f"Apify actor {actor} error: {err}")
        items = data.get("items")
        if isinstance(items, list):
            return [i for i in items if isinstance(i, dict)]
    return []


def _iso_to_date(value: Any) -> Optional[str]:
    """YYYY-MM-DD from an ISO timestamp string, else None."""
    if isinstance(value, str) and len(value) >= 10:
        return value[:10]
    return None


def _clamp(n: Any) -> int:
    """Engagement counts: coerce to non-negative int (Apify uses -1 for unknown)."""
    try:
        return max(0, int(n))
    except (ValueError, TypeError):
        return 0


def _filter_and_sort(
    items: List[Dict[str, Any]],
    from_date: str,
    to_date: str,
    source_label: str,
) -> List[Dict[str, Any]]:
    """Mirror tiktok.py date semantics: hard-filter to range, keep all if none match."""
    in_range = [i for i in items if i.get("date") and from_date <= i["date"] <= to_date]
    if in_range:
        dropped = len(items) - len(in_range)
        if dropped:
            _log(f"{source_label}: filtered {dropped} items outside date range")
        items = in_range
    else:
        _log(f"{source_label}: no items within date range, keeping all {len(items)}")
    items.sort(key=lambda x: x.get("engagement", {}).get("views", 0), reverse=True)
    return items


# ---------------------------------------------------------------------------
# TikTok (clockworks/tiktok-scraper)
# ---------------------------------------------------------------------------

def _map_tiktok_item(raw: Dict[str, Any], core_topic: str) -> Dict[str, Any]:
    """Map a clockworks/tiktok-scraper dataset item to tiktok.py's item shape."""
    video_id = str(raw.get("id", ""))
    text = raw.get("text") or ""

    author_meta = raw.get("authorMeta") if isinstance(raw.get("authorMeta"), dict) else {}
    author_name = author_meta.get("name") or ""
    # The author's OWN audience, which the actor returns on every item and
    # which was being thrown away. Without it a creator can only be ranked by
    # one post's likes — so a small account with a single decent video
    # outranks an artist with two million followers and a quiet week.
    author_fans = author_meta.get("fans")
    author_verified = bool(author_meta.get("verified"))

    url = raw.get("webVideoUrl") or ""
    if not url and author_name and video_id:
        url = f"https://www.tiktok.com/@{author_name}/video/{video_id}"

    date_str = _iso_to_date(raw.get("createTimeISO"))
    if not date_str and raw.get("createTime"):
        try:
            date_str = dates.timestamp_to_date(int(raw["createTime"]))
        except (ValueError, TypeError):
            pass

    hashtag_names = [
        h.get("name", "") for h in (raw.get("hashtags") or [])
        if isinstance(h, dict) and h.get("name")
    ]
    video_meta = raw.get("videoMeta") if isinstance(raw.get("videoMeta"), dict) else {}

    return {
        "video_id": video_id,
        "author_fans": author_fans,
        "author_verified": author_verified,
        "author_nickname": author_meta.get("nickName") or "",
        "text": text,
        "url": url,
        "author_name": author_name,
        "date": date_str,
        "engagement": {
            "views": _clamp(raw.get("playCount")),
            "likes": _clamp(raw.get("diggCount")),
            "comments": _clamp(raw.get("commentCount")),
            "shares": _clamp(raw.get("shareCount")),
        },
        "hashtags": hashtag_names,
        "duration": video_meta.get("duration"),
        "relevance": _compute_relevance(core_topic, text, hashtag_names),
        "why_relevant": f"TikTok: {text[:60]}" if text else f"TikTok: {core_topic}",
        "caption_snippet": "",
    }


def search_tiktok_apify(
    topic: str,
    from_date: str,
    to_date: str,
    depth: str = "default",
    token: str = None,
    hashtags: List[str] | None = None,
    creators: List[str] | None = None,
) -> Dict[str, Any]:
    """TikTok search via clockworks/tiktok-scraper.

    Combines search queries, hashtags, and creator profiles into a single
    actor run to pay only one actor start.
    """
    if not token:
        return {"items": [], "error": "No APIFY_API_TOKEN configured"}

    from .tiktok import _extract_core_subject, expand_tiktok_queries

    limit = DEPTH_LIMITS.get(depth, DEPTH_LIMITS["default"])
    core_topic = _extract_core_subject(topic)

    input_data: Dict[str, Any] = {"resultsPerPage": limit}
    # Plain keyword variants only: clockworks does not support OR syntax.
    # An empty topic expands to [""], and a blank searchQueries entry makes the
    # actor sweep at random. A profile-only read asks with no topic.
    queries = [
        q for q in expand_tiktok_queries(topic, depth)
        if q and q.strip() and " OR " not in q
    ]
    if queries:
        input_data["searchQueries"] = queries
        input_data["searchSection"] = "/video"
    if hashtags:
        input_data["hashtags"] = list(hashtags)
    if creators:
        input_data["profiles"] = list(creators)

    _log(f"TikTok actor run: queries={queries or '-'} hashtags={hashtags or '-'} "
         f"profiles={creators or '-'} limit={limit}")

    try:
        raw_items = _run_actor(TIKTOK_ACTOR, input_data, token)
    except Exception as e:
        _log(f"TikTok actor error: {e}")
        return {"items": [], "error": f"{type(e).__name__}: {e}"}

    seen: set[str] = set()
    items: List[Dict[str, Any]] = []
    for raw in raw_items:
        item = _map_tiktok_item(raw, core_topic)
        vid = item["video_id"]
        if vid and vid not in seen:
            seen.add(vid)
            items.append(item)

    items = _filter_and_sort(items, from_date, to_date, "TikTok")
    _log(f"TikTok: {len(items)} videos via Apify")
    return {"items": items}


# ---------------------------------------------------------------------------
# Instagram (apify/instagram-hashtag-scraper + instagram-post-scraper)
# ---------------------------------------------------------------------------

def _map_instagram_item(raw: Dict[str, Any], core_topic: str) -> Dict[str, Any]:
    """Map an Instagram actor dataset item to instagram.py's item shape."""
    post_id = str(raw.get("id", ""))
    shortcode = raw.get("shortCode") or ""
    text = raw.get("caption") or ""

    url = raw.get("url") or ""
    if not url and shortcode:
        url = f"https://www.instagram.com/p/{shortcode}"

    raw_tags = raw.get("hashtags") or []
    hashtags = [t for t in raw_tags if isinstance(t, str)]

    views = raw.get("videoPlayCount") or raw.get("videoViewCount") or 0

    return {
        "video_id": post_id or shortcode,
        "text": text,
        "url": url,
        "author_name": raw.get("ownerUsername") or "",
        "date": _iso_to_date(raw.get("timestamp")),
        "engagement": {
            "views": _clamp(views),
            "likes": _clamp(raw.get("likesCount")),
            "comments": _clamp(raw.get("commentsCount")),
        },
        "hashtags": hashtags,
        "duration": raw.get("videoDuration"),
        "relevance": _compute_relevance(core_topic, text, hashtags),
        "why_relevant": f"Instagram: {text[:60]}" if text else f"Instagram: {core_topic}",
        "caption_snippet": "",
    }


def search_instagram_apify(
    topic: str,
    from_date: str,
    to_date: str,
    depth: str = "default",
    token: str = None,
    ig_creators: List[str] | None = None,
    hashtags: List[str] | None = None,
) -> Dict[str, Any]:
    """Instagram search via Apify: hashtag scrape of the topic plus optional
    creator profile posts."""
    if not token:
        return {"items": [], "error": "No APIFY_API_TOKEN configured"}

    from .instagram import _extract_core_subject, _to_hashtag_form

    limit = DEPTH_LIMITS.get(depth, DEPTH_LIMITS["default"])
    core_topic = _extract_core_subject(topic)

    seen: set[str] = set()
    items: List[Dict[str, Any]] = []
    last_error = None

    # Every resolved hashtag, not one derived from the topic.
    #
    # Deriving glues the question into a tag nobody uses: "Armenian comedians
    # on Instagram and TikTok" became #armeniacomediansoninstagramtiktok,
    # which has no posts, so the lane reported "no results" as though there
    # were no Armenian comedians. The resolver already returns real tags —
    # armeniancomedians, armeniacomedy, comedyarmenia — and the actor takes a
    # LIST, so asking for all of them costs the same single run as asking for
    # the first. Which mattered, because the resolver orders them most
    # specific first, and the most specific tag is the likeliest to be empty.
    tags = [
        str(t).lstrip("#").strip()
        for t in (hashtags or [])
        if t and str(t).strip()
    ]
    if not tags:
        derived = _to_hashtag_form(core_topic)
        tags = [derived] if derived else []
    if tags:
        _log(f"Instagram hashtag actor run: {['#'+t for t in tags]} limit={limit}")
        try:
            raw_items = _run_actor(
                IG_HASHTAG_ACTOR,
                {"hashtags": tags, "resultsType": "posts", "resultsLimit": limit},
                token,
            )
        except Exception as e:
            _log(f"Instagram hashtag actor error: {e}")
            raw_items = []
            last_error = f"{type(e).__name__}: {e}"
        for raw in raw_items:
            item = _map_instagram_item(raw, core_topic)
            vid = item["video_id"]
            if vid and vid not in seen:
                seen.add(vid)
                items.append(item)

    if ig_creators:
        _log(f"Instagram posts actor run: creators={ig_creators}")
        try:
            raw_items = _run_actor(
                IG_POSTS_ACTOR,
                {"username": list(ig_creators), "resultsLimit": min(limit, 12)},
                token,
            )
        except Exception as e:
            _log(f"Instagram posts actor error: {e}")
            raw_items = []
            last_error = last_error or f"{type(e).__name__}: {e}"
        for raw in raw_items:
            item = _map_instagram_item(raw, core_topic)
            vid = item["video_id"]
            if vid and vid not in seen:
                seen.add(vid)
                items.append(item)

    if not items:
        return {"items": [], "error": last_error}

    items = _filter_and_sort(items, from_date, to_date, "Instagram")
    _log(f"Instagram: {len(items)} posts via Apify")
    return {"items": items, "error": last_error}
