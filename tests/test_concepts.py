"""Concepts — the kinds of people a brief track's sources describe.

Only what the sources say: every cited line is checked against the source it
names, and only a brief track's turn builds them.
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

from app.models.domain import AgentContext, Concept, ConceptEvidence, MarketEntry, TaxonomyEntry, WebFinding
from app.services import concepts as concepts_service
from app.services.concepts import build_concepts, parse_concepts
from tests.plans import discovery_plan, success_result

SOURCES = [
    WebFinding(title="Survivors recall", url="https://a.example/1", snippet="",
               content="An Ethiopian teenager accepted a promised job in Bangkok and was taken to Myanmar."),
    WebFinding(title="Scam compounds", url="https://b.example/2", snippet="",
               content="Recruiters posted Facebook adverts offering $1,500 a month and free accommodation."),
]


def _concept(**over):
    row = {
        "title": "Young Ethiopian Jobseekers in Bangkok",
        "age_range": "teenagers", "occupation": "", "job_ads": "Facebook adverts offering $1,500 a month",
        "market": "Thailand", "address": "Bangkok", "ethnicity": "Ethiopian", "home_country": "Ethiopia",
        "motivations": "a promised job", "interests": "", "boundaries": "",
        "evidence": [
            {"source": 1, "quote": "accepted a promised job in Bangkok"},
            {"source": 2, "quote": "Recruiters posted Facebook adverts"},
        ],
        "creator_search": "Ethiopian jobseekers in Bangkok discussing scams",   # ignored
    }
    row.update(over)
    return row


def test_a_concept_keeps_its_fields_and_checked_evidence():
    out = parse_concepts({"concepts": [_concept()]}, SOURCES)
    assert len(out) == 1
    c = out[0]
    assert (c.ethnicity, c.market, c.address, c.age_range) == ("Ethiopian", "Thailand", "Bangkok", "teenagers")
    assert c.job_ads.startswith("Facebook adverts")
    assert c.occupation == "" and c.interests == ""   # no source informed them
    assert [e.source for e in c.evidence] == [1, 2]
    # Built from the fields, never the model's sentence.
    assert c.creator_search == "Find Ethiopian content creators and influencers in Ethiopia, aged teenagers"


def test_a_quote_matches_despite_punctuation_and_case():
    row = _concept(evidence=[{"source": 2, "quote": "recruiters posted FACEBOOK adverts — offering $1,500 a month"}])
    assert len(parse_concepts({"concepts": [row]}, SOURCES)[0].evidence) == 1


def test_an_invented_or_misattributed_quote_is_dropped():
    row = _concept(evidence=[
        {"source": 1, "quote": "she was promised a nursing job in Kuala Lumpur"},   # invented
        {"source": 1, "quote": "Recruiters posted Facebook adverts"},              # wrong source
        {"source": 9, "quote": "accepted a promised job in Bangkok"},              # no such source
        {"source": 2, "quote": "free accommodation"},                              # real
    ])
    c = parse_concepts({"concepts": [row]}, SOURCES)[0]
    assert [(e.source, e.quote) for e in c.evidence] == [(2, "free accommodation")]


def test_a_concept_with_no_checked_evidence_is_dropped():
    rows = [
        _concept(evidence=[{"source": 1, "quote": "nothing like this is in the source"}]),
        _concept(evidence=[]),
        _concept(title=""),
        "not a concept",
    ]
    assert parse_concepts({"concepts": rows}, SOURCES) == []


def test_the_model_is_given_numbered_sources():
    sent = {}

    def create(**kw):
        sent.update(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=json.dumps({"concepts": [_concept()]})))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with patch("app.services.concepts.OpenAI", return_value=client):
        out = build_concepts("q", "Group A", SOURCES, openai_key="sk", model="m")
    user = sent["messages"][1]["content"]
    assert "The group: Group A" in user
    assert "[1] Survivors recall — https://a.example/1" in user
    assert "[2] Scam compounds" in user
    assert len(out) == 1


def test_no_sources_or_a_failed_call_builds_nothing():
    assert build_concepts("q", "G", [], openai_key="sk", model="m") == []
    with patch("app.services.concepts.OpenAI", side_effect=RuntimeError("down")):
        assert build_concepts("q", "G", SOURCES, openai_key="sk", model="m") == []


def test_only_what_the_sources_say_is_in_the_prompt():
    prompt = concepts_service.CONCEPTS_SYSTEM
    assert "ORDINARY KIND OF PERSON" in prompt
    assert "never a category of work" in prompt   # "menial jobs" is not a person
    assert "creator_search" not in prompt   # written in code, not by the model
    assert "ONLY WHAT THE SOURCES SAY" in prompt
    assert "COPIED EXACTLY" in prompt
    assert "Never invent one to reach a number" in prompt


# ── only a brief track's turn builds them ───────────────────────────


def _search_turn():
    from app.services.grounding import WebContext

    return WebContext(
        action="search", query="q",
        findings=[WebFinding(title="T", url="https://a.example/1", snippet="s", content="short")],
        source_texts=SOURCES,
    )


def _run(concepts_for):
    from app.services.agent import generate_research_plan

    ctx = AgentContext(markets=[MarketEntry(code="TH", iso="TH", name="Thailand")],
                       taxonomy=[TaxonomyEntry(slug="travel", aliases=[])])
    settings = SimpleNamespace(grounding_escalation_model="gpt-4o")
    made = [Concept(title="c", evidence=[ConceptEvidence(source=1, quote="x" * 10)])]
    extra = [WebFinding(title="Extra", url="https://c.example/3", snippet="", content="x" * 600)]
    with patch("app.services.agent.gather_web_context", return_value=_search_turn()), \
         patch("app.services.concepts.country_searches", return_value=["Country A search"]) as plan, \
         patch("app.services.concepts.more_sources", return_value=extra) as more, \
         patch("app.services.concepts.build_concepts", return_value=made) as build:
        result = generate_research_plan("q", ctx, openai_key="sk", settings=settings,
                                        concepts_for=concepts_for)
    return result, build, plan, more


def test_a_track_turn_builds_concepts_from_the_full_sources_and_country_searches():
    result, build, plan, more = _run("Group A")
    assert [c.title for c in result.concepts] == ["c"]
    args = build.call_args
    assert args.args[1] == "Group A"
    # The full text of the search's sources, then the country searches' pages.
    assert args.args[2][:2] == SOURCES and args.args[2][2].url == "https://c.example/3"
    assert more.call_args.args[0] == ["Country A search"]
    # The extra pages are shown after the turn's own, so evidence numbers match.
    assert [f.url for f in result.findings] == ["https://a.example/1", "https://c.example/3"]


def test_an_ordinary_turn_builds_none_and_searches_nothing_more():
    result, build, plan, more = _run(None)
    assert result.concepts is None
    build.assert_not_called()
    plan.assert_not_called()
    more.assert_not_called()
    assert [f.url for f in result.findings] == ["https://a.example/1"]


@patch("app.api.v1.endpoints.research.generate_research_plan")
def test_the_endpoint_asks_for_concepts_only_on_a_track_and_stores_them(mock_gen, client):
    from app.models.domain import AgentResult
    from app.services import conversations as conversation_service

    made = Concept(title="c", evidence=[ConceptEvidence(source=1, quote="x" * 10)])
    mock_gen.return_value = AgentResult(ok=True, clarifying_question="Next?", understood_so_far="x",
                                        concepts=[made])
    body = client.post("/ask", json={"prompt": "q", "track": "Group A"}).json()
    assert mock_gen.call_args.kwargs["concepts_for"] == "Group A"
    assert body["concepts"][0]["title"] == "c"
    stored = json.loads(conversation_service.store.history(body["conversation_id"], "local-dev")[-1].content)
    assert stored["concepts"][0]["title"] == "c"

    mock_gen.return_value = success_result(discovery_plan(), "discovery")
    client.post("/ask", json={"prompt": "ordinary question"})
    assert mock_gen.call_args.kwargs["concepts_for"] is None


def test_country_searches_come_from_the_model_deduplicated_and_capped():
    rows = ["Eritrean asylum seekers job scam Malaysia", "eritrean asylum seekers job scam malaysia"] + [f"s{i}" for i in range(20)]
    with patch("app.services.concepts.OpenAI", return_value=SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kw: SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=json.dumps({"countries": ["X"], "searches": rows})))]))))):
        out = concepts_service.country_searches("q", "G", openai_key="sk", model="m")
    assert out[0] == "Eritrean asylum seekers job scam Malaysia"
    assert len(out) == concepts_service.MAX_COUNTRY_SEARCHES
    with patch("app.services.concepts.OpenAI", side_effect=RuntimeError("down")):
        assert concepts_service.country_searches("q", "G", openai_key="sk", model="m") == []


def test_more_sources_skips_pages_already_found_and_pages_too_thin():
    from app.services.search.base import SearchResult
    from app.services.search.tavily import TavilyProvider

    found = {
        "a": [SearchResult(url="https://a.example/1", title="dup", content="y" * 900),
              SearchResult(url="https://n.example/new", title="new", content="y" * 900)],
        "b": [SearchResult(url="https://t.example/thin", title="thin", content="short"),
              SearchResult(url="https://n.example/new", title="again", content="y" * 900)],
    }
    with patch.object(TavilyProvider, "search", lambda self, q: found[q.text]):
        out = concepts_service.more_sources(["a", "b"], SOURCES, search_api_key="k")
    assert [f.url for f in out] == ["https://n.example/new"]
    assert concepts_service.more_sources([], SOURCES, search_api_key="k") == []
    assert concepts_service.more_sources(["a"], SOURCES, search_api_key="") == []


def test_concepts_name_the_country_not_the_region():
    assert "BE SPECIFIC" in concepts_service.CONCEPTS_SYSTEM
    assert "one search per country" in concepts_service.COVERAGE_SYSTEM


def test_find_creators_is_worded_as_a_search_for_creators():
    """Worded as a labour question, the web search returned job boards."""
    assert concepts_service.creator_search("Ethiopian", "Tech workers", "25-30", "Ethiopia") == \
        "Find Ethiopian content creators and influencers in Ethiopia who are tech workers, aged 25-30"
    assert concepts_service.creator_search("Somali", "", "30-35", "Somalia") == \
        "Find Somali content creators and influencers in Somalia, aged 30-35"
    assert concepts_service.creator_search("", "", "", "") == "Find content creators and influencers"


def _people(occupation):
    return Concept(title="t", ethnicity="Somali", age_range="30-35", home_country="Somalia",
                   occupation=occupation, evidence=[ConceptEvidence(source=1, quote="x" * 10)],
                   creator_search=concepts_service.creator_search("Somali", occupation, "30-35", "Somalia"))


def _says(payload):
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kw: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=json.dumps(payload)))]))))


def test_a_job_that_is_not_a_trade_is_dropped_from_the_search():
    """ "occupation: asylum seekers" is a situation, not people's work."""
    concepts = [_people("drivers"), _people("asylum seekers"), _people("")]
    with patch("app.services.concepts.OpenAI", return_value=_says({"trades": [True, False]})):
        out = concepts_service.keep_only_trades(concepts, openai_key="sk", model="m")
    assert out[0].creator_search == "Find Somali content creators and influencers in Somalia who are drivers, aged 30-35"
    assert out[1].occupation == ""
    assert out[1].creator_search == "Find Somali content creators and influencers in Somalia, aged 30-35"
    assert out[2] is concepts[2]          # nothing to check


def test_a_failed_or_mismatched_trade_check_changes_nothing():
    concepts = [_people("asylum seekers")]
    with patch("app.services.concepts.OpenAI", side_effect=RuntimeError("down")):
        assert concepts_service.keep_only_trades(concepts, openai_key="sk", model="m") == concepts
    with patch("app.services.concepts.OpenAI", return_value=_says({"trades": [False, True]})):
        assert concepts_service.keep_only_trades(concepts, openai_key="sk", model="m") == concepts


def test_concepts_are_the_groups_own_people_never_victims():
    prompt = concepts_service.CONCEPTS_SYSTEM
    assert "ONLY THE GROUP'S PEOPLE" in prompt
    assert 'never "victims"' in prompt


def _from(home):
    return Concept(title=f"People from {home}", home_country=home,
                   evidence=[ConceptEvidence(source=1, quote="x" * 10)])


def test_concepts_from_outside_the_group_are_dropped_in_code():
    """Told "only the group's people", the model still wrote a Kenyan concept."""
    concepts = [_from("Ethiopia"), _from("Kenya"), _from("somalia"), _from(""), _from("Ethiopia (Oromia)")]
    kept = concepts_service.from_the_group(concepts, ["Ethiopia", "Eritrea", "Somalia", "Djibouti"])
    assert [c.home_country for c in kept] == ["Ethiopia", "somalia", "", "Ethiopia (Oromia)"]
    # A group that names no place filters nothing.
    assert concepts_service.from_the_group(concepts, []) == concepts


def test_the_groups_countries_come_from_the_model_and_reach_the_concepts():
    calls = []

    def create(**kw):
        calls.append(kw["messages"])
        content = (json.dumps({"countries": ["Country A", "country a", "Country B"]})
                   if len(calls) == 1 else json.dumps({"concepts": [_concept(home_country="Country Z")]}))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with patch("app.services.concepts.OpenAI", return_value=client):
        out = build_concepts("q", "Group A", SOURCES, openai_key="sk", model="m")
    assert "The group's people are from: Country A, Country B" in calls[1][1]["content"]
    assert out == []          # its only concept was from Country Z
    with patch("app.services.concepts.OpenAI", side_effect=RuntimeError("down")):
        assert concepts_service.group_countries("G", openai_key="sk", model="m") == []
    assert "strict, narrow sense" in concepts_service.GROUP_COUNTRIES_SYSTEM
