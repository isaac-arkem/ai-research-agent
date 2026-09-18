from app.models.domain import AgentContext, MarketEntry, TaxonomyEntry
from app.services.known_accounts import (
    KnownAccount,
    extract_handles,
    handles_needing_platform,
    accounts_are_references,
    continues_named_account_job,
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


def test_a_basis_question_is_not_a_question_about_accounts():
    """Asking what "similar" means is a question about the SEARCH. Its reply —
    "same level of fame" — names nobody, so it read as a bare field answer
    inside a named-account job: the operator picked how to compare and was
    asked which platform to scrape."""
    import json
    from app.models.domain import ChatTurn

    def thread(missing):
        return [
            ChatTurn(role="user",
                     content="Find creators similar to @kevinhart4real and @billburr"),
            ChatTurn(role="assistant", content=json.dumps(
                {"clarifying_question": "q", "missing_fields": missing})),
        ]

    assert not continues_named_account_job("same level of fame", thread(["basis"]))

    # Questions that ARE about the accounts still hold the job open.
    assert continues_named_account_job("tech_giants", thread(["niche"]))
    assert continues_named_account_job("instagram", thread(["platform"]))
    # A mixed question is still about the accounts.
    assert continues_named_account_job("instagram", thread(["basis", "platform"]))


def test_an_account_given_as_a_reference_is_not_the_job():
    """"Use @demibagby and @antonielokhorst on TikTok as references to find
    similar fitness creators in Brazil" hands over two real accounts and asks
    for OTHER people. It was read as a settled scrape job and went straight to
    a plan for those two — and the escalation that would have caught it bailed
    out precisely because an "@" was present."""
    ref = ("Use @demibagby and @antonielokhorst on TikTok as references "
           "to find similar fitness creators in Brazil.")
    assert accounts_are_references(ref)
    assert not names_accounts(ref)
    assert handles_needing_platform(ref, []) == []

    for phrasing in ("find creators like @demibagby",
                     "use @demibagby as a reference to find similar creators",
                     "@demibagby and @antonielokhorst as examples, who else?"):
        assert accounts_are_references(phrasing), phrasing
        assert not names_accounts(phrasing), phrasing

    # The same accounts, actually named as the job.
    assert names_accounts("scrape @demibagby and @antonielokhorst")
