from app.models.domain import AgentContext, ChatTurn, MarketEntry, TaxonomyEntry
from app.services.prompt import (
    assemble_system_prompt,
    build_chat_messages,
    build_user_message,
)
from app.services.validator import validate_research_plan
from app.utils.guards import sanitize_prompt


def test_user_message_fences_prompt():
    text = build_user_message("Find creators in SA")
    assert '"""' in text
    assert "Find creators in SA" in text
    assert "append the country" in text


def test_chat_messages_are_openai_turns():
    history = [
        ChatTurn(role="user", content="Find fashion in SA"),
        ChatTurn(role="assistant", content="Plan for SA modest fashion."),
    ]
    messages = build_chat_messages("You are a planner.", "Instagram only", history)
    assert messages[0] == {"role": "system", "content": "You are a planner."}
    assert messages[1] == {"role": "user", "content": "Find fashion in SA"}
    assert messages[2]["role"] == "assistant"
    assert messages[-1]["role"] == "user"
    assert "Instagram only" in messages[-1]["content"]


def test_sanitize_strips_and_caps():
    assert sanitize_prompt("  hello   there  ") == "hello there"
    assert sanitize_prompt("   ") == ""


def test_system_prompt_does_not_suggest_substitute_markets():
    ctx = AgentContext(
        markets=[MarketEntry(code="SA", iso="SA", name="Saudi Arabia")],
        country_aliases={"SA": ["KSA"]},
        taxonomy=[TaxonomyEntry(slug="fashion_beauty", aliases=["fashion"])],
    )
    prompt = assemble_system_prompt(ctx)
    assert "suggest the nearest" not in prompt.lower()
    assert "Do not suggest, substitute, or invent another market." in prompt
    assert "UNSUPPORTED MARKET HANDLING:" in prompt
    assert '"recommended_runs": []' in prompt
    assert '"handles": ["isaac", "ernest"]' in prompt
    assert "Never emit one object per handle, per platform, or per niche" in prompt
    assert "handle_platforms" in prompt


def test_empty_plan_json_is_valid_for_unsupported_market():
    raw = {
        "summary": "That market is not supported.",
        "assumptions": [],
        "recommended_runs": [],
        "reference_accounts": [],
        "patterns_to_watch": [],
        "content_angles": [],
        "risks": [
            "Qatar is not in the supported markets list. I can only plan "
            "scrapes for listed markets — I will not suggest a substitute country."
        ],
    }
    result = validate_research_plan(raw, valid_country_codes={"SA", "UAE", "KW"})
    assert result.valid is True
    assert result.plan is not None
    assert result.plan.recommended_runs == []


def test_validator_rejects_unsupported_country_in_a_run():
    raw = {
        "summary": "Find creators in Qatar.",
        "assumptions": [],
        "recommended_runs": [
            {
                "pipeline": "creator_intelligence",
                "countries": ["QA"],
                "platforms": ["tiktok"],
                "hashtags": ["fashion", "style", "ootd"],
                "niche": "fashion_beauty",
                "max_creators": 50,
                "posts_per_source": 25,
                "recency_days": None,
                "title": "Qatar fashion",
                "rationale": "Operator asked for Qatar.",
            }
        ],
        "reference_accounts": [],
        "patterns_to_watch": ["Posting cadence."],
        "content_angles": ["Outfit breakdowns."],
        "risks": ["Volume may be low."],
    }
    result = validate_research_plan(raw, valid_country_codes={"SA", "UAE", "KW"})
    assert result.valid is False
    assert any(err.rule == 1 and "QA" in err.message for err in result.errors)
