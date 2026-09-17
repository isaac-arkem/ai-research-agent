from app.models.domain import AgentContext, MarketEntry, TaxonomyEntry
from app.services.known_accounts import (
    KnownAccount,
    extract_handles,
    handles_needing_platform,
    accounts_are_references,
    lookup_known_accounts,
    names_accounts,
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


def test_extracts_at_handles_and_scrape_names():
    assert extract_handles("Scrape @isaac and @ernest") == ["isaac", "ernest"]
    assert extract_handles("scrape isaac and ernest") == ["isaac", "ernest"]
    assert extract_handles(
        "Scrape the accounts @ayo_arm_media and @ernest for content related to heritage"
    ) == ["ayo_arm_media", "ernest"]
    assert extract_handles("Find modest fashion in SA") == []


def test_asking_to_find_handles_names_no_accounts():
    """A request to FIND someone's accounts must not be read as naming them.

    "scrape sarkodie's specific profiles so find his handles" used to return
    ['specific', 'so', 'find', 'his'] — the walk skipped tokens it could not
    parse and kept collecting. Four English words became accounts, which
    bypasses research, and the conversation then asked for the handles it had
    just been asked to find, once per turn, forever.
    """
    assert extract_handles(
        "yes I am looking to scrape sarkodie's specific profiles so find his handles"
    ) == []
    assert extract_handles("scrape the profiles of popular ghanaian musicians") == []
    assert extract_handles("scrape profiles for me please") == []
    assert not names_accounts("can you get me the instagram handles of sarkodie?")
    assert not names_accounts("i want you to get me the handle of sarkodie")
    # ...while an actual list of names is still a named-account job
    assert names_accounts("scrape isaac and dave")
    assert names_accounts("scrape @cookingwithnada")


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
    assert "KNOWN ACCOUNTS ALREADY IN THE CATALOG" in prompt
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


def test_a_comparison_names_seeds_not_a_job():
    """"Find creators similar to @sarkodie and @shattawale" asks for OTHER
    people. Read as a named-account job it never reached the search at all —
    it went straight to a plan to scrape the two accounts being compared
    against, and the platform question stalled it on a field that gates
    nothing."""
    cmp_ = (
        "Find creators similar to @sarkodie and @shattawale, "
        "then rank them by similarity and engagement rate."
    )
    assert accounts_are_references(cmp_)
    assert not names_accounts(cmp_)
    assert handles_needing_platform(cmp_, []) == []

    for phrasing in ("creators like @sarkodie", "lookalikes for @sarkodie",
                     "accounts comparable to @sarkodie", "competitors of @sarkodie",
                     "in the style of @sarkodie", "who resembles @sarkodie"):
        assert accounts_are_references(phrasing), phrasing
        assert not names_accounts(phrasing), phrasing

    # ...and a real job is still a real job
    for job in ("scrape @sarkodie and @shattawale", "Scrape @khloekardashian on Instagram",
                "@isaac and @dave", "scrape isaac and dave"):
        assert not accounts_are_references(job), job
        assert names_accounts(job), job


def test_would_like_is_not_a_comparison():
    """"like" carries the comparison, but not in "would like to"."""
    assert not accounts_are_references("I would like to scrape @isaac")
    assert names_accounts("I would like to scrape @isaac")
