"""Tavily provider — the one we use.

It returns page CONTENT rather than links, so the text arrives with the
result and nothing downstream has to retrieve pages.

Two quirks this pins: geo-targeting is by country NAME rather than ISO code,
and a quota error has to be distinguishable from a bad key.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.services.search import SearchError, SearchQuery, build_provider
from app.services.search.tavily import (
    MAX_RESULTS,
    TavilyProvider,
    _collect_results,
)

BODY = {
    "query": "fitness nigeria",
    "results": [
        {"url": "https://a.com", "title": "A", "content": "snippet A",
         "raw_content": "# full page A", "score": 0.9},
        {"url": "https://b.com", "title": "B", "content": "snippet B",
         "raw_content": None, "score": 0.7},
    ],
    "usage": {"credits": 1},
}


def _responding(status=200, body=None):
    resp = MagicMock()
    resp.status_code = status
    resp.text = "error body"
    resp.json.return_value = body if body is not None else BODY
    return TavilyProvider(api_key="tvly-test"), resp


def _q(**kw):
    return SearchQuery(text=kw.pop("text", "fitness"), **kw)


# ── the request ──────────────────────────────────────────────────────


def test_geo_targeting_uses_the_country_name_not_the_iso_code():
    """Tavily's country enum is names — "nigeria", not "NG". The names come
    from apify_supported_countries rather than a hardcoded table in here."""
    body = TavilyProvider("k")._body(_q(country="NG", country_name="Nigeria"))
    assert body["country"] == "nigeria"


def test_an_iso_code_alone_does_not_geo_target():
    """Sending "ng" would be rejected — better to omit than to send garbage."""
    assert "country" not in TavilyProvider("k")._body(_q(country="NG"))


def test_our_window_letters_pass_straight_through():
    for w in ("d", "w", "m", "y"):
        assert TavilyProvider("k")._body(_q(window=w))["time_range"] == w


def test_an_unknown_window_is_omitted():
    assert "time_range" not in TavilyProvider("k")._body(_q(window="fortnight"))


def test_page_content_is_requested_by_default():
    """Page text is the reason this provider was chosen, so it is on by default."""
    assert TavilyProvider("k")._body(_q())["include_raw_content"] == "markdown"
    off = TavilyProvider("k", include_content=False)._body(_q())
    assert "include_raw_content" not in off


def test_search_depth_defaults_to_advanced():
    """advanced costs 2 credits instead of 1, and returns several relevant
    snippets per page rather than one. The extra snippets are what the
    creator extraction reads, so the second credit buys the answer."""
    assert TavilyProvider("k")._body(_q())["search_depth"] == "advanced"


def test_a_cheaper_depth_is_one_setting_away():
    assert TavilyProvider("k", search_depth="basic")._body(_q())["search_depth"] == "basic"


def test_an_unknown_depth_fails_at_construction_not_as_an_http_400():
    """Tavily answers a bad depth with an opaque 400. Better to say which
    values exist, at the point the typo was made."""
    with pytest.raises(SearchError, match="unknown tavily search_depth"):
        TavilyProvider("k", search_depth="thorough")


def test_max_results_is_clamped_to_the_api_range():
    assert TavilyProvider("k")._body(_q(limit=500))["max_results"] == MAX_RESULTS
    assert TavilyProvider("k")._body(_q(limit=0))["max_results"] == 1


# ── parsing ──────────────────────────────────────────────────────────


def test_snippet_and_full_page_are_both_kept():
    """description is the relevance-selected snippet, content the whole page,
    so a caller can use either one."""
    p, resp = _responding()
    with patch("app.services.search.tavily.httpx.post", return_value=resp):
        out = p.search(_q())
    assert out[0].description == "snippet A"
    assert out[0].content == "# full page A"
    assert out[1].content is None  # raw_content was null


def test_rows_missing_a_url_or_title_are_dropped():
    body = {"results": [
        {"url": "https://a.com"}, {"title": "B"},
        {"url": "https://c.com", "title": "C"},
    ]}
    assert [r.url for r in _collect_results(body, 10)] == ["https://c.com"]


def test_an_unrecognised_envelope_yields_nothing_rather_than_raising():
    assert _collect_results({}, 5) == []
    assert _collect_results({"results": "nonsense"}, 5) == []
    assert _collect_results({"results": ["not-a-dict"]}, 5) == []


# ── failing usefully ─────────────────────────────────────────────────


def test_quota_errors_say_they_are_quota_errors():
    """429/432/433 all mean "out of credits or going too fast". The message
    has to say so — it is the only thing that tells an operator whether to
    top up or to wait."""
    for status in (429, 432, 433):
        p, resp = _responding(status=status)
        with patch("app.services.search.tavily.httpx.post", return_value=resp):
            with pytest.raises(SearchError) as caught:
                p.search(_q())
        message = str(caught.value).lower()
        assert "rate limit" in message or "quota" in message or "429" in message, status


def test_a_bad_key_says_so_and_is_not_a_rate_limit():
    """Topping up credits will not fix a wrong key, so the two must not read
    the same."""
    p, resp = _responding(status=401)
    with patch("app.services.search.tavily.httpx.post", return_value=resp):
        with pytest.raises(SearchError, match="TAVILY_API_KEY") as caught:
            p.search(_q())
    message = str(caught.value).lower()
    assert "rate limit" not in message and "quota" not in message


def test_a_missing_key_fails_before_any_request():
    with patch("app.services.search.tavily.httpx.post") as post:
        with pytest.raises(SearchError, match="TAVILY_API_KEY is not set"):
            TavilyProvider(api_key=None).search(_q())
    post.assert_not_called()


def test_a_rejected_request_shows_why():
    """400 usually means a bad parameter — most likely an unsupported country."""
    p, resp = _responding(status=400)
    with patch("app.services.search.tavily.httpx.post", return_value=resp):
        with pytest.raises(SearchError, match="HTTP 400"):
            p.search(_q())


def test_a_genuinely_empty_result_set_is_not_an_error():
    p, resp = _responding(body={"results": []})
    with patch("app.services.search.tavily.httpx.post", return_value=resp):
        assert p.search(_q()) == []


# ── canary ───────────────────────────────────────────────────────────


def test_health_catches_a_changed_response_shape():
    p, resp = _responding(body={"unexpected": True})
    with patch("app.services.search.tavily.httpx.post", return_value=resp):
        h = p.health()
    assert h.ok is False and "response shape" in h.detail


def test_health_reports_how_many_pages_came_with_content():
    p, resp = _responding()
    with patch("app.services.search.tavily.httpx.post", return_value=resp):
        h = p.health()
    assert h.ok is True
    assert "with_page_content=1" in h.checks


def test_health_reports_a_missing_key_without_calling_out():
    assert "TAVILY_API_KEY" in TavilyProvider(api_key="").health().detail


# ── the only provider ────────────────────────────────────────────────


def test_tavily_is_what_the_seam_builds():
    """The registry is gone: there is no name to pass and nothing to mistype.
    Adding a second provider later is a code change here, not an env var."""
    assert build_provider(api_key="k").name == "tavily"
