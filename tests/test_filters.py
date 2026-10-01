"""The operator's filters, read as data: what is kept, what is refused."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.services.filters import Filters, extract_filters, parse_filters


def _settings(**kw):
    base = dict(openai_api_key="sk-test", filters_model="gpt-4o", search_timeout=15.0)
    base.update(kw)
    return SimpleNamespace(**base)


def _model_says(text):
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))])
    return client


def test_only_well_formed_values_are_kept():
    f = parse_filters({
        "countries": ["sa", "SAU", "null", "GB"], "exclude_countries": ["gh"],
        "min_followers": "10,000", "max_followers": "100000", "platforms": ["TikTok", "youtube"],
        "exclude_accounts": ["@OmayaZein", "null", ""], "window_days": "30",
        "requirements": ["no brands or shops", "  ", "none"],
    })
    assert f.countries == ["SA", "GB"] and f.exclude_countries == ["GH"]
    assert f.min_followers == 10_000 and f.max_followers == 100_000
    assert f.platforms == ["tiktok"]
    assert f.exclude_accounts == ["omayazein"]
    assert f.window_days == 30 and f.requirements == ["no brands or shops"]


def test_nonsense_numbers_are_no_limit_not_a_guess():
    f = parse_filters({"max_followers": "lots", "min_followers": -5, "window_days": True})
    assert f.max_followers is None and f.min_followers is None and f.window_days is None


def test_no_filters_is_empty_and_serialises_to_nothing():
    assert Filters().is_empty() and Filters().to_dict() == {}
    assert parse_filters("not a dict").is_empty()


def test_the_conversation_reaches_the_model_and_a_failure_is_none():
    client = _model_says('{"countries": ["GB"], "max_followers": 100000}')
    history = [SimpleNamespace(role="user", content="smaller UK artists under 100k"),
               SimpleNamespace(role="assistant", content='{"clarifying_question": "Which platform?"}')]
    with patch("app.services.filters.OpenAI", return_value=client):
        f = extract_filters("instagram", history, settings=_settings())
    assert f.countries == ["GB"] and f.max_followers == 100_000
    sent = client.chat.completions.create.call_args.kwargs
    assert sent["model"] == "gpt-4o"
    assert "smaller UK artists" in sent["messages"][1]["content"]
    assert "Which platform?" in sent["messages"][1]["content"]
    with patch("app.services.filters.OpenAI", side_effect=RuntimeError("down")):
        assert extract_filters("x", [], settings=_settings()) is None


def test_reading_is_off_when_disabled_and_runs_beside_the_search_when_on():
    from app.services import grounding

    off = SimpleNamespace(filters_shadow_enabled=False, filters_apply_enabled=False)
    assert grounding._read_filters_meanwhile("x", [], {}, off) is None
    on = SimpleNamespace(filters_shadow_enabled=True, filters_apply_enabled=True)
    with patch("app.services.filters.extract_filters", return_value=Filters(max_followers=5)):
        job = grounding._read_filters_meanwhile("x", [], {}, on)
        assert grounding._filters_from(job).max_followers == 5
    assert grounding._filters_from(None) is None


def test_a_failed_reading_is_no_filters_not_an_error():
    from app.services import grounding

    on = SimpleNamespace(filters_shadow_enabled=True, filters_apply_enabled=True)
    with patch("app.services.filters.extract_filters", side_effect=RuntimeError("down")):
        assert grounding._filters_from(grounding._read_filters_meanwhile("x", [], {}, on)) is None


# ── applying the filters that data can check ────────────────────────


def _c(handle, followers=None, why=""):
    from app.models.domain import Creator
    return Creator(name=handle, handle=handle, platform="tiktok", why=why, followers=followers)


def test_excluded_accounts_are_removed_from_the_results():
    """"not @omayazein or @ayshaharun" still listed both — exclusions were
    only ever kept out of the seeds."""
    from app.services.grounding import _apply_data_filters

    kept = _apply_data_filters([_c("omayazein", 2_800_000), _c("keep", 900), _c("AyshaHarun", 600_000)],
                               Filters(exclude_accounts=["@omayazein", "ayshaharun"]))
    assert [c.handle for c in kept] == ["keep"]


def test_the_follower_range_is_checked_against_the_scraped_count():
    from app.services.grounding import _apply_data_filters

    kept = _apply_data_filters(
        [_c("big", 120_000), _c("ok", 48_600), _c("tiny", 300), _c("text_only", None, "75,000 followers on tiktok")],
        Filters(min_followers=1_000, max_followers=50_000))
    assert [c.handle for c in kept] == ["ok"]


def test_an_unknown_size_is_kept_and_labelled_only_when_a_size_was_asked_for():
    from app.services.grounding import _apply_data_filters

    kept = _apply_data_filters([_c("mystery", None, "named on a blog")], Filters(max_followers=50_000))
    assert kept[0].why == "follower count unknown — named on a blog"
    again = _apply_data_filters(kept, Filters(max_followers=50_000))
    assert again[0].why.count("unknown") == 1
    plain = _apply_data_filters([_c("mystery", None, "named on a blog")], Filters(countries=["SA"]))
    assert plain[0].why == "named on a blog"


def test_no_filters_change_nothing():
    from app.services.grounding import _apply_data_filters

    creators = [_c("a", 5), _c("b", None, "x")]
    assert _apply_data_filters(creators, Filters()) == creators


def test_scraped_creators_carry_their_real_follower_count():
    from app.services.grounding import creators_from_post_authors

    item = SimpleNamespace(source="tiktok", author="hayabyishma", url="u", title="", body="b",
                           snippet="", engagement={"likes": 10},
                           metadata={"author_fans": 184_900, "author_verified": False,
                                     "author_nickname": "", "hashtags": []})
    assert creators_from_post_authors([item])[0].followers == 184_900


# ── step 3: the judged filters reach the relevance filter ───────────


def _ctx_with(*pairs):
    from app.models.domain import AgentContext, MarketEntry
    return AgentContext(markets=[MarketEntry(code=c, iso=c, name=n) for c, n in pairs], taxonomy=[])


def test_no_filters_leaves_the_relevance_request_exactly_as_before():
    from app.services.grounding import _filters_to_check

    assert _filters_to_check(None, None) == ""
    assert _filters_to_check(Filters(max_followers=5, exclude_accounts=["x"]), None) == ""


def test_countries_and_requirements_become_a_checklist_in_names():
    from app.services.grounding import _filters_to_check

    text = _filters_to_check(
        Filters(countries=["SA"], exclude_countries=["PK"], requirements=["no brands or shops"]),
        _ctx_with(("SA", "Saudi Arabia"), ("PK", "Pakistan")))
    assert "Must be based in: Saudi Arabia" in text
    assert "Must NOT be based in: Pakistan" in text
    assert "- no brands or shops" in text
    assert "no evidence is not a failure" in text


def test_only_the_operators_requirements_are_labelled_never_a_country():
    """A country label went on all 81 cards of a Saudi search."""
    from app.services.grounding import _label_unverified

    creators = [_c("a", 5, "x"), _c("b", 5, "y")]
    rows = [{"n": 1, "what": "based in Saudi Arabia"}, {"n": 2, "what": "No brands or shops"}]
    out = _label_unverified(creators, rows, requirements=["no brands or shops"])
    assert out[0].why == "x"
    assert out[1].why == "unverified: no brands or shops — y"
    assert _label_unverified(creators, rows, requirements=[]) == creators


def test_a_long_list_is_judged_in_batches_and_drops_land_on_the_right_creators():
    """One call over a hundred and twenty accounts drifted; thirty at a time
    does not — and a drop in the second batch must hit the second batch."""
    from app.services import grounding

    creators = [_c(f"c{i}", 100) for i in range(65)]
    sent = []

    def model(*a, **k):
        client = MagicMock()

        def create(**kw):
            listed = kw["messages"][1]["content"].split("Accounts:\n", 1)[1]
            sent.append(listed.count("\n") + 1)
            first = listed.split("(@", 1)[1].split(")", 1)[0]
            drop = '{"drop": [{"n": 2, "why": "shop"}]}' if first == "c30" else '{"drop": []}'
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=drop))])
        client.chat.completions.create.side_effect = create
        return client

    with patch("app.services.grounding.OpenAI", side_effect=model):
        kept = grounding.drop_irrelevant_creators(creators, "q", openai_key="sk")
    assert sorted(sent) == [5, 30, 30]
    assert "c31" not in [c.handle for c in kept] and len(kept) == 64


def test_a_short_list_is_still_one_call():
    from app.services import grounding

    with patch("app.services.grounding.OpenAI", return_value=_model_says('{"drop": []}')) as made:
        grounding.drop_irrelevant_creators([_c("a", 1), _c("b", 1)], "q", openai_key="sk")
    assert made.return_value.chat.completions.create.call_count == 1


# ── step 4: filters ride with the answer into the next message ──────


def test_the_latest_applied_filters_are_found_in_the_thread():
    import json
    from app.services.grounding import _filters_last_applied

    history = [
        SimpleNamespace(role="assistant", content=json.dumps({"filters": {"countries": ["GB"]}})),
        SimpleNamespace(role="user", content="more"),
        SimpleNamespace(role="assistant", content=json.dumps({"filters": {"max_followers": 50000}})),
        SimpleNamespace(role="assistant", content="plain text"),
    ]
    assert _filters_last_applied(history) == {"max_followers": 50000}
    assert _filters_last_applied([]) is None


def test_the_reader_is_shown_what_was_applied_last():
    client = _model_says('{"countries": ["GB"]}')
    with patch("app.services.filters.OpenAI", return_value=client):
        extract_filters("look for more", [], settings=_settings(),
                        previous={"countries": ["GB"], "max_followers": 50000})
    content = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert 'Filters applied to the last answer: {"countries": ["GB"], "max_followers": 50000}' in content


def test_a_search_answer_stores_its_filters():
    import json
    from app.api.v1.endpoints import research
    from app.models.domain import AgentResult, WebFinding

    saved = []
    store = SimpleNamespace(add_assistant_message=lambda cid, uid, content, **k: saved.append(content) or "m")
    result = AgentResult(ok=True, clarifying_question="next?", understood_so_far="found",
                         findings=[WebFinding(title="t", url="https://x", snippet="s")],
                         filters={"countries": ["SA"], "max_followers": 50000})
    with patch.object(research.conversation_service, "store", store):
        research._persist_assistant("c", "u", result)
    assert json.loads(saved[0])["filters"] == {"countries": ["SA"], "max_followers": 50000}
