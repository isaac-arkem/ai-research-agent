"""Unsupported countries are dropped, not fatal.

An operator asking for "MENA" should get the MENA countries we can scrape,
not a validation error because the region also contains Jordan.
"""

from app.services.country_filter import filter_unsupported_countries

VALID = {"SA", "AE", "EG", "MA", "NG"}


def _plan(**over):
    base = {
        "summary": "MENA fitness discovery.",
        "assumptions": ["Standard defaults applied."],
        "recommended_runs": [{
            "pipeline": "creator_intelligence",
            "countries": ["SA", "AE", "JO", "QA"],
            "platforms": ["tiktok"],
            "hashtags": ["a", "b", "c"],
            "niche": "fitness", "max_creators": 50, "posts_per_source": 25,
            "title": "MENA fitness", "rationale": "Region expansion.",
        }],
        "reference_accounts": [], "patterns_to_watch": [],
        "content_angles": [], "risks": [],
    }
    base.update(over)
    return base


def test_unsupported_codes_are_removed_and_the_rest_survive():
    out, dropped = filter_unsupported_countries(_plan(), VALID)
    assert out["recommended_runs"][0]["countries"] == ["SA", "AE"]
    assert dropped == ["JO", "QA"]


def test_every_dropped_country_is_named_in_assumptions():
    """Never silent — the operator must see what was left out."""
    out, _ = filter_unsupported_countries(_plan(), VALID)
    note = out["assumptions"][-1]
    assert "JO" in note and "QA" in note
    assert "Standard defaults applied." in out["assumptions"]  # originals kept


def test_a_fully_unsupported_plan_says_so_in_risks():
    """Not a cheerful empty plan — the operator asked for something we cannot
    do and has to be told."""
    plan = _plan()
    plan["recommended_runs"][0]["countries"] = ["JO", "QA"]
    out, dropped = filter_unsupported_countries(plan, VALID)
    assert out["recommended_runs"] == []
    assert any("nothing to scrape" in r for r in out["risks"])


def test_runs_emptied_by_filtering_are_dropped_entirely():
    plan = _plan()
    plan["recommended_runs"].append({
        "pipeline": "creator_intelligence", "countries": ["JO"],
        "platforms": ["tiktok"], "hashtags": ["a", "b", "c"], "niche": "fitness",
        "max_creators": 50, "posts_per_source": 25,
        "title": "Jordan", "rationale": "x",
    })
    out, _ = filter_unsupported_countries(plan, VALID)
    assert len(out["recommended_runs"]) == 1
    assert out["recommended_runs"][0]["countries"] == ["SA", "AE"]


def test_a_clean_plan_is_returned_untouched():
    plan = _plan()
    plan["recommended_runs"][0]["countries"] = ["SA", "AE"]
    out, dropped = filter_unsupported_countries(plan, VALID)
    assert dropped == []
    assert out is plan  # no copy, no spurious assumption


def test_reference_accounts_are_left_alone():
    """FLOW 3 carries handles, not countries — nothing to filter."""
    plan = _plan(recommended_runs=[], reference_accounts=[
        {"pipeline": "reference_profiles", "handles": ["x"],
         "platforms": ["tiktok"], "niche": "fitness", "rationale": "y"}])
    out, dropped = filter_unsupported_countries(plan, VALID)
    assert dropped == []
    assert out["reference_accounts"]


def test_malformed_input_does_not_raise():
    for bad in ({}, {"recommended_runs": "nope"},
                {"recommended_runs": [{"countries": "SA"}]},
                {"recommended_runs": [{"countries": [None, 5, "SA"]}]}):
        filter_unsupported_countries(bad, VALID)
