"""Existing niches come from the database, not a Python file.

Two sources merged: `niches` is the curated table new niches are created in,
and reference_accounts.topic still carries older slugs that predate it. The
list is a HINT — the model reuses a slug when one fits and coins a new one
when none does — so a fetch failure degrades rather than breaks.
"""

from unittest.mock import MagicMock, patch

from app.core.config import get_settings
from app.core.dependencies import _fetch_taxonomy_from_db, _normalise_slug


def _client(niches=(), topics=()):
    client = MagicMock()

    def table(name):
        handle = MagicMock()
        if name == "niches":
            handle.select.return_value.eq.return_value.execute.return_value.data = list(niches)
        else:
            handle.select.return_value.execute.return_value.data = list(topics)
        return handle

    client.table.side_effect = table
    return client


def _fetch(niches=(), topics=()):
    with patch("app.core.dependencies.get_supabase_admin",
               return_value=_client(niches, topics)):
        return _fetch_taxonomy_from_db(get_settings())


# ── slug hygiene ─────────────────────────────────────────────────────


def test_uppercase_slugs_are_lowercased():
    """The plan schema requires ^[a-z0-9_]+$. Live data has "MAGA" next to
    "maga"; offering the former would fail validation if the model reused it."""
    assert _normalise_slug("MAGA") == "maga"


def test_slugs_the_schema_would_reject_are_dropped():
    assert _normalise_slug("catgirls dubai") is None   # space
    assert _normalise_slug("egypt-jordan") is None     # hyphen
    assert _normalise_slug("") is None
    assert _normalise_slug(None) is None


def test_test_rows_are_filtered_out():
    """reference_accounts.topic is usage data and carries smoke tests.
    "codex_smoke" must never reach a scrape plan as a niche."""
    for junk in ("khaby_test", "codex_smoke", "sona_test", "demo_reference"):
        assert _normalise_slug(junk) is None


# ── merging the two sources ──────────────────────────────────────────


def test_both_sources_are_merged():
    out = _fetch(
        niches=[{"slug": "catgirls_dubai", "label": "Catgirls Dubai",
                 "description": None, "active": True}],
        topics=[{"topic": "music_dance"}],
    )
    assert [e.slug for e in out] == ["catgirls_dubai", "music_dance"]


def test_the_niches_table_wins_on_a_clash():
    """It is curated and carries label/description; topic is just usage."""
    out = _fetch(
        niches=[{"slug": "cooking_mum", "label": "Cooking Mum",
                 "description": "Home cooks", "active": True}],
        topics=[{"topic": "cooking_mum"}],
    )
    assert len(out) == 1
    assert out[0].aliases == ["Cooking Mum", "Home cooks"]


def test_case_variants_collapse_to_one():
    out = _fetch(topics=[{"topic": "MAGA"}, {"topic": "maga"}])
    assert [e.slug for e in out] == ["maga"]


def test_label_and_description_become_matching_hints():
    """They help the model tie "cosplay in Dubai" to catgirls_dubai."""
    out = _fetch(niches=[{"slug": "catgirls_dubai", "label": "Catgirls Dubai",
                          "description": "Cosplay creators", "active": True}])
    assert out[0].aliases == ["Catgirls Dubai", "Cosplay creators"]


def test_results_are_sorted_for_a_stable_prompt():
    out = _fetch(topics=[{"topic": "zebra"}, {"topic": "alpha"}])
    assert [e.slug for e in out] == ["alpha", "zebra"]


# ── failure policy ───────────────────────────────────────────────────


def test_a_db_failure_degrades_rather_than_raising():
    """Unlike the country list, this is a hint. Losing it means the model may
    coin a slug that already exists — not that it cannot plan."""
    client = MagicMock()
    client.table.side_effect = Exception("supabase down")
    with patch("app.core.dependencies.get_supabase_admin", return_value=client):
        assert _fetch_taxonomy_from_db(get_settings()) == []


def test_no_supabase_returns_empty_not_an_error():
    with patch("app.core.dependencies.get_supabase_admin", return_value=None):
        assert _fetch_taxonomy_from_db(get_settings()) == []
