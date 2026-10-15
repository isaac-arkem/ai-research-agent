"""News, reports and Telegram — free sources the planner may use on any
question — and the operator choosing sources by name. Paid lanes still need
the platform named."""

from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.services.filters import Filters, parse_filters
from app.services.research import orchestrator
from app.services.research.engine import news, normalize, reliefweb, telegram_public
from app.services.search.base import SearchQuery, SearchResult
from app.services.search.tavily import TavilyProvider, _collect_results, _day


# ── Tavily: new options, old requests unchanged ─────────────────────


def test_an_existing_tavily_request_is_unchanged():
    body = TavilyProvider("k")._body(SearchQuery(text="q", country_name="Ghana", window="m", limit=20))
    assert body == {
        "query": "q", "max_results": 20, "search_depth": "advanced", "topic": "general",
        "country": "ghana", "time_range": "m", "include_raw_content": "markdown",
    }


def test_the_new_tavily_options_are_sent_only_when_set():
    body = TavilyProvider("k")._body(SearchQuery(
        text="q", topic="news", start_date="2025-01-01", end_date="2025-12-31",
        include_domains=["t.me"],
    ))
    assert body["topic"] == "news"
    assert (body["start_date"], body["end_date"]) == ("2025-01-01", "2025-12-31")
    assert body["include_domains"] == ["t.me"]


@pytest.mark.parametrize("raw, day", [
    ("Wed, 19 Feb 2025 10:00:00 GMT", "2025-02-19"),
    ("2025-04-15T08:00:00Z", "2025-04-15"),
    (None, None),
    ("not a date", None),
])
def test_a_published_date_becomes_a_day(raw, day):
    assert _day(raw) == day


def test_results_keep_their_published_date():
    rows = _collect_results({"results": [
        {"url": "https://a.example/x", "title": "A", "content": "s", "published_date": "2025-02-19"},
        {"url": "https://b.example/y", "title": "B", "content": "s"},
    ]}, 5)
    assert [r.published_date for r in rows] == ["2025-02-19", None]


# ── the news lane ───────────────────────────────────────────────────


def test_news_asks_the_news_topic_inside_the_window():
    sent = {}

    def search(self, query):
        sent["query"] = query
        return [SearchResult(url="https://www.reuters.com/a", title="Survivors recall",
                             description="snippet", content="body", published_date="2025-02-19")]

    with patch.object(TavilyProvider, "search", search):
        items = news.search_news("scam centre recruitment", "2025-01-01", "2025-12-31",
                                 {"TAVILY_API_KEY": "k"})
    q = sent["query"]
    assert (q.topic, q.start_date, q.end_date) == ("news", "2025-01-01", "2025-12-31")
    assert items[0]["date"] == "2025-02-19"
    assert items[0]["source_domain"] == "reuters.com"


def test_news_items_normalize_as_dated_pages():
    item = {"id": "NW1", "title": "T", "url": "https://x.example/a", "snippet": "s",
            "date": "2025-02-19", "source_domain": "x.example"}
    out = normalize.normalize_source_items("news", [item], "2025-01-01", "2025-12-31")
    assert out[0].published_at == "2025-02-19"
    assert out[0].source == "news"
    # Outside the window it is dropped, like every other source.
    assert normalize.normalize_source_items("news", [item], "2026-01-01", "2026-12-31") == []


# ── the reports lane ────────────────────────────────────────────────


def test_reports_say_when_they_are_not_configured():
    with pytest.raises(reliefweb.ReliefWebNotConfigured):
        reliefweb.search_reports("q", "2025-01-01", "2025-12-31", {})


def test_reports_are_read_with_their_publisher_and_date():
    payload = {"data": [
        {"id": 1, "fields": {
            "title": "A Wicked Problem", "url_alias": "https://reliefweb.int/report/x",
            "date": {"created": "2026-02-20T00:00:00+00:00"},
            "source": [{"name": "OHCHR"}], "body": "Fraudulent  recruitment\n and exploitation.",
        }},
        {"id": 2, "fields": {"title": "", "url_alias": "https://reliefweb.int/report/y"}},
        {"id": 3, "fields": {"title": "No alias"}},
    ]}
    items = reliefweb.parse_reports(payload)
    assert [i["title"] for i in items] == ["A Wicked Problem", "No alias"]
    first = items[0]
    assert (first["date"], first["source_domain"]) == ("2026-02-20", "OHCHR")
    assert first["snippet"] == "Fraudulent recruitment and exploitation."
    assert items[1]["url"] == "https://reliefweb.int/node/3"


def test_reports_send_the_appname_and_window():
    response = MagicMock(status_code=200)
    response.json.return_value = {"data": []}
    with patch("app.services.research.engine.reliefweb.httpx.post", return_value=response) as post:
        reliefweb.search_reports("scam", "2025-01-01", "2025-12-31", {"RELIEFWEB_APPNAME": "app"})
    kwargs = post.call_args.kwargs
    assert kwargs["params"] == {"appname": "app"}
    assert kwargs["json"]["filter"]["value"]["from"].startswith("2025-01-01")


# ── the Telegram lane ───────────────────────────────────────────────


PREVIEW = """
<div class="tgme_channel_info_header_title"><span dir="auto">Jobs &amp; Work</span></div>
<div class="tgme_channel_info_counter"><span class="counter_value">12.5K</span> <span class="counter_type">subscribers</span></div>
<div class="tgme_widget_message_wrap js-widget_message_wrap"><div class="tgme_widget_message" data-post="jobsbkk/77">
<div class="tgme_widget_message_text js-message_text" dir="auto">High salary<br/>Free <b>accommodation</b></div>
<span class="tgme_widget_message_views">3.4K</span><time datetime="2025-06-01T10:00:00+00:00" class="time">10:00</time>
</div></div>
<div class="tgme_widget_message_wrap js-widget_message_wrap"><div class="tgme_widget_message" data-post="jobsbkk/78">
<div class="tgme_widget_message_photo_wrap"></div>
</div></div>
"""


def test_a_channel_preview_is_read_into_posts():
    items = telegram_public.parse_preview(PREVIEW, "jobsbkk")
    assert len(items) == 1  # the photo with no words is nothing to read
    post = items[0]
    assert post["text"] == "High salary\nFree accommodation"
    assert post["url"] == "https://t.me/jobsbkk/77"
    assert (post["date"], post["engagement"], post["subscribers"]) == ("2025-06-01", {"views": 3400}, 12500)
    assert post["display_name"] == "Jobs & Work"


def test_channels_come_out_of_telegram_links_only():
    assert telegram_public.channels_in([
        "https://t.me/s/jobsbkk/1234", "https://t.me/jobsbkk", "https://t.me/+AbCdEf",
        "https://t.me/joinchat/xyz", "https://t.me/cambodiajobs2026/77",
        "https://example.com/t.me/x", "https://t.me/ab",
    ]) == [("jobsbkk", "1234"), ("cambodiajobs2026", "77")]


def test_telegram_finds_channels_then_reads_them():
    found = [SearchResult(url="https://t.me/jobsbkk/77", title="t")]
    page = MagicMock(status_code=200, text=PREVIEW)
    with patch.object(TavilyProvider, "search", return_value=found) as search, \
         patch("app.services.research.engine.telegram_public.httpx.get", return_value=page) as get:
        items = telegram_public.search_telegram("jobs", "2025-01-01", "2025-12-31", {"TAVILY_API_KEY": "k"})
    assert search.call_args.args[0].include_domains == ["t.me"]
    assert get.call_args.args[0] == "https://t.me/s/jobsbkk/77"
    assert [i["handle"] for i in items] == ["jobsbkk"]


def test_a_telegram_channel_becomes_an_account_card():
    from app.services.grounding import creators_from_post_authors

    items = normalize.normalize_source_items(
        "telegram", telegram_public.parse_preview(PREVIEW, "jobsbkk"), "2025-01-01", "2025-12-31")
    creator = creators_from_post_authors(items)[0]
    assert (creator.handle, creator.platform, creator.name) == ("jobsbkk", "telegram", "Jobs & Work")
    assert creator.profile_url == "https://t.me/jobsbkk"
    assert creator.followers == 12500


# ── choosing sources ────────────────────────────────────────────────


def test_the_free_sources_are_offered_to_the_planner():
    for source in ("news", "reports", "telegram"):
        assert source in orchestrator.LANES and source in orchestrator.AVAILABLE_SOURCES
    # Paid lanes still need the platform named.
    assert not orchestrator.paid_lane_allowed("tiktok", [])
    assert orchestrator.paid_lane_allowed("news", [])
    assert orchestrator.paid_lane_allowed("telegram", [])


def test_reports_wait_for_their_setting():
    assert "reports" not in orchestrator.default_sources({})
    assert "reports" in orchestrator.default_sources({"RELIEFWEB_APPNAME": "app"})
    assert {"news", "telegram"} <= set(orchestrator.default_sources({}))


def test_a_plan_draws_on_the_default_sources(monkeypatch):
    monkeypatch.setattr(orchestrator, "default_sources", lambda config=None: ["grounding", "news"])
    plan = orchestrator.plan_for("recruitment offers in Cambodia")
    assert {s for sq in plan.subqueries for s in sq.sources} <= {"grounding", "news"}


def test_a_chosen_source_is_the_only_source():
    plan = orchestrator.plan_for("recruitment offers", sources=orchestrator.lanes_for(["telegram"]))
    assert {s for sq in plan.subqueries for s in sq.sources} == {"telegram"}


def test_source_names_map_to_lanes():
    assert orchestrator.lanes_for(["Web", "reddit", "news", "web", "facebook"]) == ["grounding", "reddit", "news"]


def test_the_filters_reader_keeps_only_known_sources():
    assert parse_filters({"sources": ["Reddit", "news", "facebook"]}).sources == ["reddit", "news"]
    # Nothing chosen stores nothing — the filters of every earlier search look the same.
    assert "sources" not in parse_filters({"platforms": ["tiktok"]}).to_dict()


def _job(value):
    future = Future()
    future.set_result(value)
    return future


@pytest.mark.parametrize("found, enabled, lanes", [
    (Filters(sources=["web", "reddit"]), True, ["grounding", "reddit"]),
    (Filters(countries=["GH"]), True, []),
    (None, True, []),
    (Filters(sources=["reddit"]), False, []),
])
def test_the_search_uses_the_sources_the_operator_chose(found, enabled, lanes):
    from app.services.grounding import _chosen_lanes

    settings = SimpleNamespace(filters_apply_enabled=enabled)
    assert _chosen_lanes(_job(found), settings) == lanes
    assert _chosen_lanes(None, settings) == []


def test_an_unconfigured_source_is_recorded_not_fatal():
    subquery = SimpleNamespace(search_query="q")
    source, _sq, items, outcome = orchestrator._run_lane(
        "reports", subquery, "2025-01-01", "2025-12-31", "default", {}, {})
    assert items == [] and outcome.state == "error"
    assert "RELIEFWEB_APPNAME" in outcome.detail


def test_news_and_telegram_run_once_per_request(monkeypatch):
    from app.services.research.engine import schema

    calls = []
    for lane in ("news", "telegram", "reddit"):
        monkeypatch.setitem(orchestrator.LANES, lane,
                            lambda q, *a, _lane=lane, **k: calls.append(_lane) or [])
    plan = orchestrator.plan_for("q", sources=["news", "telegram", "reddit"])
    plan.subqueries = [
        schema.SubQuery(label=f"s{i}", search_query=f"q{i}", ranking_query="q",
                        sources=["news", "telegram", "reddit"], weight=1.0)
        for i in range(3)
    ]
    result = orchestrator.run_research(topic="q", plan=plan, config={},
                                       window=("2025-01-01", "2025-12-31"))
    assert sorted(calls) == ["news", "reddit", "reddit", "reddit", "telegram"]
    skipped = [o for o in result.lane_outcomes if o.state == "skipped"]
    assert {o.source for o in skipped} == {"news", "telegram"}


def test_telegram_keeps_the_posts_closest_to_the_question():
    posts = [
        {"text": "weekend football results", "date": "2025-06-03"},
        {"text": "chat operator job high salary free accommodation", "date": "2025-06-01"},
        {"text": "job vacancy accommodation provided", "date": "2025-06-02"},
    ]
    kept = telegram_public.closest_posts(posts, "job high salary accommodation", 2)
    assert [p["text"] for p in kept] == [
        "chat operator job high salary free accommodation",
        "job vacancy accommodation provided",
    ]
    assert kept[0]["relevance"] >= kept[1]["relevance"]


def test_hacker_news_and_polymarket_run_only_when_named():
    defaults = orchestrator.default_sources({"RELIEFWEB_APPNAME": "app"})
    assert "hackernews" not in defaults and "polymarket" not in defaults
    assert "hackernews" not in orchestrator.SOURCES_FOR_ANSWER["markets"]
    # Naming them still works.
    assert orchestrator.lanes_for(["hackernews", "polymarket"]) == ["hackernews", "polymarket"]
    plan = orchestrator.plan_for("q", sources=["hackernews"])
    assert {s for sq in plan.subqueries for s in sq.sources} == {"hackernews"}




def test_news_searches_the_operators_question():
    with patch("app.services.research.orchestrator.news.search_news", return_value=[]) as search:
        orchestrator._lane_news("kw", "2025-01-01", "2025-12-31", "default", {},
                                {"raw_topic": "recruitment of locals"})
    assert search.call_args.args[0] == "recruitment of locals"
