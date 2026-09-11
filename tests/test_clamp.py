"""Out-of-range numbers are corrected, not refused.

An operator asking for 5,000 posts an account has made an ordinary mistake.
Before this, three different things could happen: the model capped it, the
model asked them to pick a smaller number, or the validator rejected the plan
and the turn died with a 422. The first is right, and it was a judgment call
that drifted — 900 was capped, 5,000 was questioned.

So the correction is deterministic now, and every correction is written into
assumptions where the operator can see it and disagree.
"""

from app.services.clamp import (
    MAX_CREATORS_CHOICES,
    POSTS_MAX,
    clamp_parameters,
)


def _run(**kw):
    job = {"pipeline": "creator_intelligence", "countries": ["NG"],
           "platforms": ["tiktok"], "niche": "fitness",
           "max_creators": 50, "posts_per_source": 25}
    job.update(kw)
    return {"recommended_runs": [job], "reference_accounts": []}


def _ref(**kw):
    job = {"pipeline": "reference_profiles", "handles": ["isaac"],
           "platforms": ["tiktok"], "niche": "tech_giants", "posts_per_source": 10}
    job.update(kw)
    return {"recommended_runs": [], "reference_accounts": [job]}


# ── posts ────────────────────────────────────────────────────────────


def test_too_many_posts_is_capped_at_the_maximum():
    plan, notes = clamp_parameters(_run(posts_per_source=5000))
    assert plan["recommended_runs"][0]["posts_per_source"] == POSTS_MAX == 100
    assert "100" in notes[0] and "5000" in notes[0]


def test_zero_posts_is_raised_to_one():
    """A scrape of nothing is not a scrape."""
    plan, notes = clamp_parameters(_run(posts_per_source=0))
    assert plan["recommended_runs"][0]["posts_per_source"] == 1
    assert "below the minimum" in notes[0]


def test_a_value_in_range_is_left_alone_and_unremarked():
    plan, notes = clamp_parameters(_run(posts_per_source=25))
    assert plan["recommended_runs"][0]["posts_per_source"] == 25
    assert notes == []


def test_named_account_jobs_are_capped_too():
    """reference_accounts carry their own post count."""
    plan, notes = clamp_parameters(_ref(posts_per_source=900))
    assert plan["reference_accounts"][0]["posts_per_source"] == POSTS_MAX
    assert "per account" in notes[0]


# ── creators ─────────────────────────────────────────────────────────


def test_an_unlisted_creator_count_moves_to_the_nearest_allowed_one():
    plan, notes = clamp_parameters(_run(max_creators=75))
    assert plan["recommended_runs"][0]["max_creators"] == 100
    assert "75 is not one of" in notes[0]


def test_a_tie_rounds_up():
    """Asked for more than a listed value, the operator more likely wants the
    larger of the two than the smaller."""
    plan, _ = clamp_parameters(_run(max_creators=15))   # 10 and 20 are equidistant
    assert plan["recommended_runs"][0]["max_creators"] == 20


def test_every_allowed_count_survives_untouched():
    for choice in MAX_CREATORS_CHOICES:
        plan, notes = clamp_parameters(_run(max_creators=choice))
        assert plan["recommended_runs"][0]["max_creators"] == choice
        assert notes == []


# ── what it refuses to guess at ──────────────────────────────────────


def test_a_non_number_is_left_for_the_validator():
    """"lots" is not an out-of-range number, it is a wrong kind of value —
    and inventing one to replace it would hide a real fault."""
    plan, notes = clamp_parameters(_run(posts_per_source="lots"))
    assert plan["recommended_runs"][0]["posts_per_source"] == "lots"
    assert notes == []


def test_a_numeric_string_is_still_a_number():
    plan, _ = clamp_parameters(_run(posts_per_source="5000"))
    assert plan["recommended_runs"][0]["posts_per_source"] == POSTS_MAX


def test_a_missing_value_is_not_invented():
    plan, notes = clamp_parameters({"recommended_runs": [{"niche": "fitness"}]})
    assert "posts_per_source" not in plan["recommended_runs"][0]
    assert notes == []


def test_a_plan_with_no_jobs_is_handled():
    plan, notes = clamp_parameters({})
    assert notes == []


def test_the_posts_limit_agrees_where_it_is_enforced():
    """The clamp and the prompt must state the same number, or a plan clamps
    cleanly and then fails validation anyway."""
    import re

    from app.services.prompt import PARAMETER_LIMITS

    stated = re.search(r"posts_per_source: integer between 1 and (\d+)", PARAMETER_LIMITS)
    assert stated and int(stated.group(1)) == POSTS_MAX == 100


def test_the_stored_model_is_deliberately_looser_than_the_limit():
    """It parses history as well as new plans. A plan written when the limit
    was 200 was valid then, and tightening this bound only made old threads
    fail to open — it enforces nothing, because clamp.py and validator.py
    already stop a new plan going over."""
    from app.models.domain import RecommendedRun

    field = RecommendedRun.model_fields["posts_per_source"]
    ceiling = next(m.le for m in field.metadata if hasattr(m, "le"))
    assert ceiling > POSTS_MAX

    # an old plan still loads
    from tests.plans import discovery_plan

    stored = discovery_plan().model_dump()
    stored["recommended_runs"][0]["posts_per_source"] = 200   # legal when written
    from app.models.domain import ResearchPlan

    ResearchPlan.model_validate(stored)
