"""What the Instagram lane actually asks Apify for."""

from unittest.mock import patch

from app.services.research.engine import apify_social


def _payloads(**kwargs):
    sent = []

    def fake(actor, payload, token, fields=None):
        sent.append(payload)
        return []

    with patch.object(apify_social, "_run_actor", side_effect=fake):
        apify_social.search_instagram_apify(
            "Armenian comedians on Instagram and TikTok",
            "2025-01-01", "2026-01-01", token="x", **kwargs,
        )
    return sent


def test_resolved_hashtags_are_used_instead_of_one_glued_from_the_topic():
    """Deriving a tag from the question produced
    #armeniancomediansoninstagramtiktok, which has no posts — so the lane
    reported "no results" as though there were no Armenian comedians. It even
    swallowed the platform names into the tag."""
    sent = _payloads(hashtags=["armeniancomedians", "armeniacomedy", "comedyarmenia"])

    assert sent[0]["hashtags"] == [
        "armeniancomedians", "armeniacomedy", "comedyarmenia"
    ]
    # The actor takes a list, so every tag costs the same single run as one.
    assert len(sent) == 1


def test_a_leading_hash_is_accepted():
    sent = _payloads(hashtags=["#armeniancomedy", " comedyarmenia "])
    assert sent[0]["hashtags"] == ["armeniancomedy", "comedyarmenia"]


def test_without_resolved_tags_it_still_derives_one():
    """The resolver returns nothing when the engine has no reasoning client,
    and the lane has to keep working — worse, but working."""
    sent = _payloads()
    assert sent[0]["hashtags"] == ["armeniancomediansoninstagramtiktok"]


def test_empty_hashtags_fall_back_rather_than_asking_for_nothing():
    sent = _payloads(hashtags=["", "   ", None])
    assert sent[0]["hashtags"] == ["armeniancomediansoninstagramtiktok"]


def test_only_the_fields_the_mapper_reads_are_asked_for():
    """run-sync-get-dataset-items streams the dataset in the same response
    that runs the actor, and retrying is not an option — a retried run bills
    again. So a response that dies mid-download is data paid for and lost.

    It does die: two profiles returned 473,832 bytes and then IncompleteRead,
    which the lane honestly reported as "could not read @_zinatubako" and the
    hunt fell back to adjectives. 272,894 bytes carried 28 fields per item to
    read 10; asking for the ten returns the same posts in 8,867."""
    asked = []

    def fake(actor, payload, token, fields=None):
        asked.append(fields)
        return []

    with patch.object(apify_social, "_run_actor", side_effect=fake):
        apify_social.search_instagram_apify(
            "armenian comedians", "2025-01-01", "2026-01-01",
            token="x", hashtags=["armeniancomedians"], ig_creators=["someone"],
        )

    assert asked and all(f for f in asked), "a lane asked for every field"
    for f in asked:
        for needed in ("ownerUsername", "caption", "hashtags", "timestamp",
                       "likesCount", "url"):
            assert needed in f, needed


def test_discovery_does_not_call_the_people_search_actor():
    """apify/instagram-search-scraper fails most of the time. A named country
    used to switch the lane onto it and skip the hashtag scrape, which is how
    a request for both platforms came back TikTok-only."""
    import inspect

    source = inspect.getsource(apify_social)
    assert "instagram-search-scraper" not in source
    assert not hasattr(apify_social, "IG_SEARCH_ACTOR")


# ── Instagram's hashtags are weighed before they are scraped ────────

def test_only_tags_worth_scraping_are_kept_and_only_from_the_list():
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch
    from app.services.grounding import _instagram_tags_worth_scraping

    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content='{"scrape": ["#EthiopianTech", "invented", "somaliart"]}'))]
    client = MagicMock(); client.chat.completions.create.return_value = resp
    settings = SimpleNamespace(openai_api_key="sk", grounding_model="m", search_timeout=5)
    with patch("app.services.grounding.OpenAI", return_value=client):
        kept = _instagram_tags_worth_scraping(["ethiopiantech", "programming", "somaliart"], "q", settings=settings)
    assert kept == ["ethiopiantech", "somaliart"]
    with patch("app.services.grounding.OpenAI", side_effect=RuntimeError("down")):
        assert _instagram_tags_worth_scraping(["programming"], "q", settings=settings) == []


def test_the_instagram_scrape_uses_the_weighed_tags_and_tiktok_does_not():
    from unittest.mock import patch
    from app.services.research import orchestrator

    opts = {"hashtags": ["ethiopiantech", "programming"], "instagram_hashtags": ["ethiopiantech"]}
    with patch("app.services.research.orchestrator.instagram.search_and_enrich", return_value={"items": []}) as ig, \
         patch("app.services.research.orchestrator.instagram.parse_instagram_response", return_value=[]):
        orchestrator._lane_instagram("q", "2025-01-01", "2025-12-31", "default", {}, opts)
    assert ig.call_args.kwargs["hashtags"] == ["ethiopiantech"]
    with patch("app.services.research.orchestrator.tiktok.search_and_enrich", return_value={"items": []}) as tt, \
         patch("app.services.research.orchestrator.tiktok.parse_tiktok_response", return_value=[]):
        orchestrator._lane_tiktok("q", "2025-01-01", "2025-12-31", "default", {}, opts)
    assert tt.call_args.kwargs["hashtags"] == ["ethiopiantech", "programming"]
    # Without a weighed list, Instagram gets the shared one, as before.
    with patch("app.services.research.orchestrator.instagram.search_and_enrich", return_value={"items": []}) as ig, \
         patch("app.services.research.orchestrator.instagram.parse_instagram_response", return_value=[]):
        orchestrator._lane_instagram("q", "2025-01-01", "2025-12-31", "default", {}, {"hashtags": ["a", "b"]})
    assert ig.call_args.kwargs["hashtags"] == ["a", "b"]
