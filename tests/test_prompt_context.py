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
    brief = "situation. " * 400  # ~4400, used to be over the old 2000 cap
    assert len(brief) > 2000
    assert sanitize_prompt(brief) == brief.strip()


def test_system_prompt_does_not_suggest_substitute_markets():
    ctx = AgentContext(
        markets=[MarketEntry(code="SA", iso="SA", name="Saudi Arabia")],
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


def test_followup_message_says_edit_the_most_recent_plan():
    """Regression: the revise-the-previous-plan instruction used to live only in
    the first-message branch, so it was dropped on exactly the turns that needed
    it — and the model edited an older plan instead of the newest one."""
    text = build_user_message("change the niche to baking", is_followup=True)
    assert "the most recent one" in text
    assert "superseded" in text
    assert "originally asked" not in text


def test_followup_message_still_handles_clarifying_answers():
    """The same branch serves both follow-up kinds — fixing plan edits must not
    regress 'you asked which platform, I said TikTok'."""
    text = build_user_message("TikTok", is_followup=True)
    assert "clarifying question" in text
    assert "treat this reply as the answer" in text


def test_add_a_country_rule_reaches_followup_turns():
    """'also add Nigeria' is always a follow-up, so the merge rule must be in
    both branches — it used to be in the first-message branch only."""
    for followup in (False, True):
        assert "append the country" in build_user_message("also add NG", is_followup=followup)


def test_followup_branch_is_the_one_used_when_history_exists():
    """Pins the wiring: history present => follow-up text is what gets sent."""
    history = [
        ChatTurn(role="user", content="find fitness creators in saudi"),
        ChatTurn(role="assistant", content='{"summary": "Fitness in SA"}'),
    ]
    messages = build_chat_messages("SYS", "change the niche", history)
    assert "the most recent one" in messages[-1]["content"]

    no_history = build_chat_messages("SYS", "find creators in SA", [])
    assert "the most recent one" not in no_history[-1]["content"]


def _guarded_prompt():
    ctx = AgentContext(
        markets=[MarketEntry(code="SA", iso="SA", name="Saudi Arabia")],
        taxonomy=[TaxonomyEntry(slug="fashion_beauty", aliases=["fashion"])],
    )
    return assemble_system_prompt(ctx)


def test_conversation_guard_is_in_the_system_prompt():
    prompt = _guarded_prompt()
    assert "CONVERSATION GUARD" in prompt
    assert "GREETING / SMALL TALK" in prompt
    assert "MEANINGLESS OR UNINTELLIGIBLE" in prompt
    assert "INSTRUCTION-OVERRIDE ATTEMPT" in prompt


def test_guard_forbids_echoing_the_previous_plan():
    """The reported bug: 'hello' / '123' / 'forget all instructions' came back
    as the last plan repeated verbatim."""
    prompt = _guarded_prompt()
    assert "never repeat a previous plan unchanged" in prompt.lower()
    assert "forget all instructions" in prompt


def test_followup_triages_before_assuming_an_edit():
    """A plan existing in history must not turn every later message into an edit."""
    text = build_user_message("hello", is_followup=True)
    assert "CONVERSATION GUARD" in text
    assert "Never hand back the previous plan just because one exists." in text
    # the edit path must be explicitly conditional, not the default
    assert text.index("FIRST, apply the CONVERSATION GUARD") < text.index("ONLY IF it is a genuine edit")


def test_edit_target_skips_empty_offtopic_replies():
    """An off-topic reply is an empty plan — it must not become the edit target."""
    text = build_user_message("change the niche", is_followup=True)
    assert "actually has runs or" in text
    assert "skip past any empty off-topic replies" in text


def test_the_prompt_says_what_assumptions_are_for():
    """It used to say only "each inference you made", and the model filled it
    with a readback of the request — "The operator wants to scrape the TikTok
    account of @isaac" — which tells the operator nothing they could disagree
    with. The point of the field is catching a decision before it runs."""
    from app.models.domain import AgentContext, MarketEntry
    from app.services.prompt import assemble_system_prompt

    prompt = assemble_system_prompt(
        AgentContext(markets=[MarketEntry(code="NG", iso="NG", name="Nigeria")],
                     taxonomy=[])
    )
    assert "ASSUMPTIONS:" in prompt
    # what it is for
    assert "catch a decision they disagree with" in prompt
    # and the shape of a wrong one, named explicitly
    assert "Never restate the request" in prompt
    assert "The operator wants to scrape the TikTok account of @isaac" in prompt
    # an empty array is a valid answer, not a gap to pad
    assert "return an empty array" in prompt
