from app.models.domain import ReferenceAccount, ResearchPlan
from app.services.validator import validate_research_plan


def _plan(**ref) -> dict:
    return {
        "summary": "Scrape named accounts.",
        "assumptions": [],
        "recommended_runs": [],
        "reference_accounts": [ref],
        "patterns_to_watch": ["Posting cadence."],
        "content_angles": ["Duet the cooking clips."],
        "risks": ["Handles may be region-blocked."],
    }


def test_same_platform_handles_share_one_entry():
    account = ReferenceAccount(
        pipeline="reference_profiles",
        handles=["isaac", "ernest"],
        platform="tiktok",
        niche="cooking_mum",
        posts_per_source=10,
        recency_days=None,
        rationale="Named by the operator.",
    )
    dumped = account.model_dump()
    assert dumped["handles"] == ["isaac", "ernest"]
    assert dumped["platforms"] == ["tiktok"]
    assert "handle" not in dumped
    assert "platform" not in dumped


def test_legacy_handle_coerces_to_handles():
    account = ReferenceAccount.model_validate(
        {
            "pipeline": "reference_profiles",
            "handle": "@isaac",
            "platform": "tiktok",
            "niche": "cooking_mum",
            "rationale": "Stored older thread.",
        }
    )
    assert account.handles == ["isaac"]
    assert account.platforms == ["tiktok"]


def test_validator_accepts_handles_array():
    result = validate_research_plan(
        _plan(
            pipeline="reference_profiles",
            handles=["isaac", "ernest"],
            platform="tiktok",
            niche="cooking_mum",
            posts_per_source=10,
            recency_days=None,
            rationale="Named by the operator.",
        ),
        valid_country_codes={"SA"},
    )
    assert result.valid is True
    assert result.plan is not None
    assert result.plan.reference_accounts[0].handles == ["isaac", "ernest"]
    assert result.plan.reference_accounts[0].platforms == ["tiktok"]


def test_validator_coerces_legacy_handle():
    result = validate_research_plan(
        _plan(
            pipeline="reference_profiles",
            handle="isaac",
            platform="tiktok",
            niche="cooking_mum",
            rationale="Named by the operator.",
        ),
        valid_country_codes={"SA"},
    )
    assert result.valid is True
    assert result.plan is not None
    assert result.plan.reference_accounts[0].handles == ["isaac"]


def test_validator_merges_same_platform_rows():
    raw = {
        "summary": "Scrape named accounts.",
        "assumptions": [],
        "recommended_runs": [],
        "reference_accounts": [
            {
                "pipeline": "reference_profiles",
                "handles": ["isaac"],
                "platform": "tiktok",
                "niche": "cooking_mum",
                "posts_per_source": 10,
                "recency_days": None,
                "rationale": "Named by the operator.",
            },
            {
                "pipeline": "reference_profiles",
                "handles": ["ernest"],
                "platform": "tiktok",
                "niche": "cooking_mum",
                "posts_per_source": 10,
                "recency_days": None,
                "rationale": "Named by the operator.",
            },
        ],
        "patterns_to_watch": ["Posting cadence."],
        "content_angles": ["Duet the cooking clips."],
        "risks": ["Handles may be region-blocked."],
    }
    result = validate_research_plan(raw, valid_country_codes={"SA"})
    assert result.valid is True
    assert result.plan is not None
    assert len(result.plan.reference_accounts) == 1
    assert result.plan.reference_accounts[0].handles == ["isaac", "ernest"]
    assert result.plan.reference_accounts[0].platforms == ["tiktok"]


def test_validator_merges_different_platforms_into_one_job():
    raw = {
        "summary": "Scrape named accounts.",
        "assumptions": [],
        "recommended_runs": [],
        "reference_accounts": [
            {
                "pipeline": "reference_profiles",
                "handles": ["isaac"],
                "platform": "tiktok",
                "niche": "cooking_mum",
                "posts_per_source": 10,
                "recency_days": None,
                "rationale": "Named by the operator.",
            },
            {
                "pipeline": "reference_profiles",
                "handles": ["ernest"],
                "platform": "instagram",
                "niche": "cooking_mum",
                "posts_per_source": 10,
                "recency_days": None,
                "rationale": "Named by the operator.",
            },
        ],
        "patterns_to_watch": ["Posting cadence."],
        "content_angles": ["Duet the cooking clips."],
        "risks": ["Handles may be region-blocked."],
    }
    result = validate_research_plan(raw, valid_country_codes={"SA"})
    assert result.valid is True
    assert result.plan is not None
    assert len(result.plan.reference_accounts) == 1
    job = result.plan.reference_accounts[0]
    assert job.handles == ["isaac", "ernest"]
    assert job.platforms == ["tiktok", "instagram"]
    assert job.handle_platforms == {"isaac": "tiktok", "ernest": "instagram"}


def test_validator_keeps_one_niche_when_rows_disagree():
    raw = {
        "summary": "Scrape named accounts.",
        "assumptions": [],
        "recommended_runs": [],
        "reference_accounts": [
            {
                "pipeline": "reference_profiles",
                "handles": ["ayo_arm_media"],
                "platform": "tiktok",
                "niche": "cooking_mum",
                "posts_per_source": 10,
                "recency_days": None,
                "rationale": "Catalog handle.",
            },
            {
                "pipeline": "reference_profiles",
                "handles": ["ernest"],
                "platform": "instagram",
                "niche": "fashion_beauty",
                "posts_per_source": 10,
                "recency_days": None,
                "rationale": "Added in this request.",
            },
        ],
        "patterns_to_watch": ["Posting cadence."],
        "content_angles": ["Duet the cooking clips."],
        "risks": ["Handles may be region-blocked."],
    }
    result = validate_research_plan(raw, valid_country_codes={"SA"})
    assert result.valid is True
    assert result.plan is not None
    job = result.plan.reference_accounts[0]
    assert job.handles == ["ayo_arm_media", "ernest"]
    assert job.niche == "cooking_mum"


def test_research_plan_collapses_split_platform_rows():
    plan = ResearchPlan(
        summary="Scrape named accounts.",
        assumptions=[],
        recommended_runs=[],
        reference_accounts=[
            ReferenceAccount(
                pipeline="reference_profiles",
                handles=["isaac"],
                platform="tiktok",
                niche="cooking_mum",
                rationale="Named by the operator.",
            ),
            ReferenceAccount(
                pipeline="reference_profiles",
                handles=["ernest"],
                platform="instagram",
                niche="cooking_mum",
                rationale="Named by the operator.",
            ),
        ],
        patterns_to_watch=["Posting cadence."],
        content_angles=["Duet the cooking clips."],
        risks=["Handles may be region-blocked."],
    )
    assert len(plan.reference_accounts) == 1
    job = plan.reference_accounts[0]
    assert job.handles == ["isaac", "ernest"]
    assert job.platforms == ["tiktok", "instagram"]
    assert job.handle_platforms == {"isaac": "tiktok", "ernest": "instagram"}


def test_validator_coerces_handle_platforms_casing():
    result = validate_research_plan(
        _plan(
            pipeline="reference_profiles",
            handles=["ayo_arm_media"],
            platforms=["TikTok"],
            handle_platforms={"ayo_arm_media": "TikTok"},
            niche="cooking_mum",
            posts_per_source=10,
            recency_days="any",
            rationale="Named by the operator.",
        ),
        valid_country_codes={"SA"},
    )
    assert result.valid is True
    assert result.plan is not None
    job = result.plan.reference_accounts[0]
    assert job.handles == ["ayo_arm_media"]
    assert job.platforms == ["tiktok"]
    assert job.handle_platforms == {"ayo_arm_media": "tiktok"}
    assert job.recency_days is None
