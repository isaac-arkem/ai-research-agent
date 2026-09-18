"""What the Instagram lane actually asks Apify for."""

from unittest.mock import patch

from app.services.research.engine import apify_social


def _payloads(**kwargs):
    sent = []

    def fake(actor, payload, token):
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
