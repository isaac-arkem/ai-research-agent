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
