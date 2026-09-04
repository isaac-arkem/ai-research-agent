from app.models.domain import (
    AgentResult,
    RecommendedRun,
    ReferenceAccount,
    ResearchPlan,
)


def discovery_plan() -> ResearchPlan:
    return ResearchPlan(
        summary="Find modest fashion creators in Saudi Arabia.",
        assumptions=["Platform defaulted to both TikTok and Instagram."],
        recommended_runs=[
            RecommendedRun(
                pipeline="creator_intelligence",
                countries=["SA"],
                platforms=["tiktok", "instagram"],
                hashtags=["modestfashion", "ازياء", "hijabstyle"],
                niche="fashion_beauty",
                max_creators=50,
                posts_per_source=25,
                recency_days=None,
                title="SA modest fashion discovery",
                rationale="Hashtag discovery in the named market.",
            )
        ],
        reference_accounts=[],
        patterns_to_watch=["Coverage of abaya vs western modest wear."],
        content_angles=["Day-to-night modest outfit breakdowns."],
        risks=["Arabic hashtag volume may be seasonal around Ramadan."],
    )


def deep_research_plan() -> ResearchPlan:
    runs = []
    for code, title in (("UAE", "UAE cooking"), ("SA", "SA cooking"), ("KW", "KW cooking")):
        runs.append(
            RecommendedRun(
                pipeline="creator_intelligence",
                countries=[code],
                platforms=["tiktok"],
                hashtags=["cooking", "وصفات", "ramadanrecipes"],
                niche="cooking_mum",
                max_creators=100,
                posts_per_source=50,
                recency_days=None,
                title=title,
                rationale="One localized run per Gulf market.",
            )
        )
    return ResearchPlan(
        summary="Compare cooking creators across the Gulf.",
        assumptions=["Gulf expanded to UAE, SA, KW."],
        recommended_runs=runs,
        reference_accounts=[],
        patterns_to_watch=["Family vs restaurant cooking."],
        content_angles=["Same recipe, three-market plating."],
        risks=["Ramadan recency will skew results."],
    )


def reference_plan() -> ResearchPlan:
    return ResearchPlan(
        summary="Scrape two named accounts.",
        assumptions=[],
        recommended_runs=[],
        reference_accounts=[
            ReferenceAccount(
                pipeline="reference_profiles",
                handles=["khloekardashian"],
                platform="instagram",
                niche="fashion_beauty",
                rationale="Named by the operator.",
            ),
            ReferenceAccount(
                pipeline="reference_profiles",
                handles=["charlidamelio"],
                platform="tiktok",
                niche="music_dance",
                rationale="Named by the operator.",
            ),
        ],
        patterns_to_watch=["Posting cadence."],
        content_angles=["Duet with the dance account."],
        risks=["Handles may be region-blocked."],
    )


def success_result(plan: ResearchPlan, flow: str) -> AgentResult:
    return AgentResult(
        ok=True,
        plan=plan,
        flow=flow,
        latency_ms=42,
        llm_latency_ms=30,
        prompt_tokens=800,
        completion_tokens=400,
    )
