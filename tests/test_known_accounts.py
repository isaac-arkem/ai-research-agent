from app.models.domain import AgentContext, MarketEntry, TaxonomyEntry
from app.services.known_accounts import (
    KnownAccount,
    accounts_shown_in_history,
    extract_handles,
    handles_needing_platform,
    accounts_are_references,
    known_accounts_from_text,
    lookup_known_accounts,
    platform_clarifying_question,
    shared_job_niche,
)
from app.services.prompt import assemble_system_prompt


class _Result:
    def __init__(self, data):
        self.data = data


class _Query:
    def __init__(self, rows):
        self.rows = rows
        self._values = []

    def select(self, *_args, **_kwargs):
        return self

    def in_(self, _column, values):
        self._values = [str(v).lower() for v in values]
        return self

    def execute(self):
        data = [
            row
            for row in self.rows
            if str(row.get("handle", "")).lower() in self._values
        ]
        return _Result(data)


class _Client:
    def __init__(self, rows):
        self.rows = rows

    def table(self, name):
        assert name == "reference_accounts"
        return _Query(self.rows)


def _ctx():
    return AgentContext(
        markets=[MarketEntry(code="SA", iso="SA", name="Saudi Arabia")],
        taxonomy=[TaxonomyEntry(slug="fashion_beauty", aliases=["fashion"])],
    )


def test_extracts_the_at_handles_and_nothing_else():
    """An @ is an @ — no judgement needed, so this is still code.

    It used to also guess at BARE names after "scrape", deciding from two
    word lists which of the following words were names. "scrape sarkodie's
    specific profiles so find his handles" became the accounts ['specific',
    'so', 'find', 'his'], which flipped the turn into a named-account job and
    swallowed the rest of the thread. Who the operator named is the router's
    "subjects" now."""
    assert extract_handles("Scrape @isaac and @ernest") == ["isaac", "ernest"]
    assert extract_handles(
        "Scrape the accounts @ayo_arm_media and @ernest for content related to heritage"
    ) == ["ayo_arm_media", "ernest"]
    assert extract_handles("Find modest fashion in SA") == []
    # The one that started it: prose is never promoted to accounts.
    assert extract_handles(
        "scrape sarkodie's specific profiles so find his handles"
    ) == []
    assert extract_handles("scrape isaac and ernest") == []


def test_lookup_returns_platform_and_niche():
    client = _Client(
        [
            {"handle": "isaac", "platform": "tiktok", "topic": "cooking_mum"},
            {"handle": "ernest", "platform": "tiktok", "topic": "cooking_mum"},
        ]
    )
    found = lookup_known_accounts(["isaac", "ernest"], client=client)
    assert {(a.handle, a.platform, a.niche) for a in found} == {
        ("isaac", "tiktok", "cooking_mum"),
        ("ernest", "tiktok", "cooking_mum"),
    }


def test_lookup_skips_rows_without_a_usable_platform():
    client = _Client(
        [{"handle": "isaac", "platform": "youtube", "topic": "cooking_mum"}]
    )
    assert lookup_known_accounts(["isaac"], client=client) == []


def test_prompt_tells_the_model_not_to_ask_for_catalog_fields():
    prompt = assemble_system_prompt(
        _ctx(),
        known_accounts=[
            KnownAccount(handle="isaac", platform="tiktok", niche="cooking_mum"),
        ],
    )
    assert "KNOWN ACCOUNTS" in prompt
    assert "Do not ask which platform they are on" in prompt
    assert "@isaac | tiktok | cooking_mum" in prompt
    assert "SHARED JOB NICHE: cooking_mum" in prompt
    assert "Apply this niche to EVERY handle in the current request" in prompt
    assert "take it from KNOWN ACCOUNTS" in prompt


def test_shared_job_niche_covers_handles_added_in_the_request():
    known = [
        KnownAccount(handle="ayo_arm_media", platform="tiktok", niche="cooking_mum"),
        KnownAccount(handle="ernest", platform="instagram", niche=None),
    ]
    assert shared_job_niche(known) == "cooking_mum"
    prompt = assemble_system_prompt(_ctx(), known_accounts=known)
    assert "SHARED JOB NICHE: cooking_mum" in prompt
    assert "do not ask for another niche" in prompt.lower()


def test_unknown_handle_needs_platform_unless_operator_named_one():
    known = [
        KnownAccount(handle="ayo_arm_media", platform="tiktok", niche="heritage_diaspora"),
        KnownAccount(handle="ayo_arm_media", platform="instagram", niche="heritage_diaspora"),
    ]
    prompt = "Scrape @ayo_arm_media and @ernest"
    assert handles_needing_platform(prompt, known=known) == ["ernest"]
    assert handles_needing_platform(
        "Scrape @ayo_arm_media and @ernest on TikTok",
        known=known,
    ) == []
    assert (
        platform_clarifying_question(["ernest"])
        == "Which platform is @ernest on — TikTok or Instagram?"
    )


def test_prompt_lists_handles_that_still_need_a_platform():
    prompt = assemble_system_prompt(
        _ctx(),
        known_accounts=[
            KnownAccount(handle="ayo_arm_media", platform="tiktok", niche="heritage_diaspora"),
        ],
        handles_needing_platform=["ernest"],
    )
    assert "HANDLES NOT IN THE CATALOG" in prompt
    assert "@ernest" in prompt
    assert "Do NOT return a ResearchPlan" in prompt


def test_picking_an_option_does_not_restart_a_named_account_job():
    """"same audience" — the whole of a reply that picks how to compare —
    names nobody. The handles it inherits come from the question two turns
    back, so read on its own it looked like a bare answer inside a
    named-account job: the platform question fired, grounding was skipped,
    and picking an option ENDED the search instead of refining it."""
    from app.models.domain import ChatTurn

    after_comparison = [
        ChatTurn(role="user",
                 content="Find creators similar to @kevinhart4real and @billburr"),
        ChatTurn(role="assistant",
                 content='{"clarifying_question":"which basis?","missing_fields":[]}'),
    ]
    for pick in ("same audience", "same storytelling style", "similar level of fame",
                 "something i typed myself"):
        assert handles_needing_platform(pick, after_comparison) == [], pick

    # A message carrying its OWN handles is still judged on its own merits, so
    # a real job later in the same thread still asks which platform.
    still_asked = handles_needing_platform("scrape @isaac and @dave", after_comparison)
    assert "isaac" in still_asked and "dave" in still_asked

    # Separately, and not changed here: that list also carries the seeds from
    # the earlier comparison, because handles are gathered across every user
    # turn in the thread. Asking which platform @kevinhart4real is on, in a
    # job about @isaac, is wrong — but it is wrong the same way it was before
    # any of this, and fixing it is a change to how a job inherits handles.
    assert "kevinhart4real" in still_asked


def test_a_handle_already_shown_with_a_platform_is_not_asked_again():
    """The planner sees eight turns of history. The Python gate that asks
    which platform @x is on did not: it only looked at what the operator
    typed, so 'scrape @mikhail_litvin' after we had just returned him on
    Instagram asked the question we had already answered."""
    import json
    from app.models.domain import ChatTurn
    from app.services.known_accounts import (
        accounts_shown_in_history,
        known_accounts_from_text,
    )

    shown = ChatTurn(
        role="assistant",
        content=json.dumps({
            "clarifying_question": "Need more details?",
            "missing_fields": [],
            "creators": [{
                "name": "Mikhail Litvin",
                "handle": "mikhail_litvin",
                "platform": "instagram",
                "why": "engaging lifestyle vlogs and collaborations",
            }],
        }),
    )
    later_ask = ChatTurn(
        role="assistant",
        content=json.dumps({
            "clarifying_question": "Which platform is @mikhail_litvin on?",
            "missing_fields": ["platform"],
        }),
    )
    history = [
        ChatTurn(role="user", content="can you get me his handle??"),
        shown,
        ChatTurn(role="user", content="scrape @mikhail_litvin"),
        later_ask,
    ]

    found = accounts_shown_in_history(history)
    assert [(a.handle, a.platform) for a in found] == [
        ("mikhail_litvin", "instagram"),
    ]

    known = known_accounts_from_text(
        "the platform is already available, it was part of the results",
        history,
        client=_Client([]),
    )
    assert any(
        a.handle == "mikhail_litvin" and a.platform == "instagram" for a in known
    )
    assert handles_needing_platform(
        "scrape @mikhail_litvin", history, known
    ) == []
    assert handles_needing_platform(
        "the platform is already available, it was part of the results",
        history, known,
    ) == []
    assert handles_needing_platform(
        "i mean scrape Mikhail Litvin, dont you already know its platform???",
        history, known,
    ) == []


def test_a_handle_never_shown_still_needs_a_platform():
    """The gate only skips the question when this thread already named one."""
    from app.models.domain import ChatTurn

    history = [
        ChatTurn(role="user", content="find cooking creators in Ghana"),
        ChatTurn(role="assistant",
                 content='{"clarifying_question":"Do these look right?",'
                         '"missing_fields":[],"creators":[]}'),
    ]
    known = known_accounts_from_text(
        "scrape @ernest", history, client=_Client([]),
    )
    assert known == []
    assert handles_needing_platform("scrape @ernest", history, known) == ["ernest"]


