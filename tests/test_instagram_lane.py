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


def _runs(**kwargs):
    sent = []

    def fake(actor, payload, token, fields=None):
        sent.append((actor, payload, fields))
        if actor == apify_social.IG_SEARCH_ACTOR:
            return [{"username": "uk_skin"}]
        return []

    with patch.object(apify_social, "_run_actor", side_effect=fake):
        apify_social.search_instagram_apify(
            "Find skincare creators in the UK on Instagram using #cleanbeauty",
            "2025-01-01", "2026-01-01", token="x", **kwargs,
        )
    return sent


def test_a_named_market_uses_user_search_not_the_hashtag_actor():
    """The hashtag feed is global. Country + keyword in one people-search
    query is how community-mapper finds market accounts without a second
    scrape of the same tag."""
    sent = _runs(
        hashtags=["cleanbeauty", "skinimalism"],
        country_name="United Kingdom",
    )

    assert [actor for actor, _, _ in sent] == [apify_social.IG_SEARCH_ACTOR]
    assert "username" in (sent[0][2] or "")
    assert "ownerUsername" not in (sent[0][2] or "")
    assert sent[0][1] == {
        "search": "United Kingdom cleanbeauty creator, United Kingdom skinimalism creator",
        "searchType": "user",
        "searchLimit": 20,
        "liveSearch": True,
    }


def test_user_search_retries_without_creator_only_when_the_first_query_is_empty():
    calls = []

    def fake(actor, payload, token, fields=None):
        calls.append(payload["search"])
        if "creator" in payload["search"]:
            return []
        return [{"username": "uk_skin", "biography": "London"}]

    with patch.object(apify_social, "_run_actor", side_effect=fake):
        result = apify_social.search_instagram_apify(
            "skincare", "2025-01-01", "2026-01-01", token="x",
            hashtags=["cleanbeauty"], country_name="United Kingdom",
        )

    assert calls == [
        "United Kingdom cleanbeauty creator",
        "United Kingdom cleanbeauty",
    ]
    assert result["items"][0]["author_name"] == "uk_skin"


def test_a_successful_user_search_does_not_run_the_fallback_or_hashtag():
    actors = []

    def fake(actor, payload, token, fields=None):
        actors.append(actor)
        return [{"username": "uk_skin", "fullName": "UK Skin", "followersCount": 1200}]

    with patch.object(apify_social, "_run_actor", side_effect=fake):
        result = apify_social.search_instagram_apify(
            "skincare", "2025-01-01", "2026-01-01", token="x",
            hashtags=["cleanbeauty"], country_name="United Kingdom",
        )

    assert actors == [apify_social.IG_SEARCH_ACTOR]
    item = result["items"][0]
    assert item["author_name"] == "uk_skin"
    assert item["author_fans"] == 1200
    assert item["url"] == "https://www.instagram.com/uk_skin/"


def test_named_handles_still_use_the_posts_actor_when_a_market_is_set():
    """Seeds they named are a profile read, not a second discovery scrape."""
    sent = _runs(
        hashtags=["cleanbeauty"],
        country_name="United Kingdom",
        ig_creators=["someone"],
    )

    assert [actor for actor, _, _ in sent] == [
        apify_social.IG_SEARCH_ACTOR,
        apify_social.IG_POSTS_ACTOR,
    ]


def test_without_a_market_the_hashtag_actor_still_runs():
    sent = _runs(hashtags=["cleanbeauty"])
    assert [actor for actor, _, _ in sent] == [apify_social.IG_HASHTAG_ACTOR]
    assert sent[0][1]["hashtags"] == ["cleanbeauty"]


def test_a_named_market_without_tags_searches_the_topic_not_a_glued_hashtag():
    sent = []

    def fake(actor, payload, token, fields=None):
        sent.append((actor, payload))
        return [{"username": "x"}]

    with patch.object(apify_social, "_run_actor", side_effect=fake):
        apify_social.search_instagram_apify(
            "Armenian comedians on Instagram and TikTok",
            "2025-01-01", "2026-01-01", token="x",
            country_name="Armenia",
        )

    assert sent[0][0] == apify_social.IG_SEARCH_ACTOR
    query = sent[0][1]["search"]
    assert "armeniancomediansoninstagramtiktok" not in query
    assert query.startswith("Armenia ")
    assert query.endswith(" creator")
