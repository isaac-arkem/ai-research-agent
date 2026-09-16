"""Which lanes a research run actually calls, and what it refuses to pay for."""

from app.services.research import orchestrator as orc
from app.services.research.engine import schema


def _plan(sources):
    sq = schema.SubQuery(
        label="handles",
        search_query="Sarkodie handles",
        ranking_query="Sarkodie handles",
        sources=list(sources),
        weight=1.0,
    )
    return schema.QueryPlan(
        intent="find",
        freshness_mode="any",
        cluster_mode="none",
        raw_topic="Sarkodie handles",
        subqueries=[sq],
        source_weights={s: 1.0 for s in sources},
    )


def _run(monkeypatch, **kwargs):
    """Run with every lane stubbed, and report which ones were called."""
    called = []
    for name in list(orc.LANES):
        def lane(*_a, _n=name, **_k):
            called.append(_n)
            return []
        monkeypatch.setitem(orc.LANES, name, lane)
    result = orc.run_research(
        topic="Sarkodie handles",
        plan=_plan(["grounding", "instagram", "tiktok", "reddit"]),
        config={},
        window=("2025-01-01", "2026-01-01"),
        **kwargs,
    )
    return called, result


def test_a_question_about_one_person_never_calls_a_paid_lane(monkeypatch):
    """"what are his handles?" returned 37 creators, 35 of them strangers who
    had posted under #sarkodie — and every one of them was paid for. His own
    handles come off the web lane, which reads his profile pages. So the
    scrape is not run and filtered, it is not run."""
    called, result = _run(monkeypatch, subject="Sarkodie",
                          force_lanes=["instagram", "tiktok"])

    assert "instagram" not in called
    assert "tiktok" not in called
    assert "grounding" in called
    # Recorded as declined, not as having found nothing.
    assert result.source_status["instagram"] == "skipped"
    assert result.source_status["tiktok"] == "skipped"


def test_a_list_question_still_scrapes_the_named_platforms(monkeypatch):
    """The subject gate must not become a second scrape gate. A list question
    naming Instagram and TikTok still queries both."""
    called, result = _run(monkeypatch, subject=None,
                          force_lanes=["instagram", "tiktok"])

    assert "instagram" in called
    assert "tiktok" in called
    assert result.source_status["instagram"] != "skipped"


def test_no_platform_named_still_means_no_scrape(monkeypatch):
    """The original cost guard is unchanged by the subject gate."""
    called, _ = _run(monkeypatch, subject=None, force_lanes=[])

    assert "instagram" not in called
    assert "tiktok" not in called
