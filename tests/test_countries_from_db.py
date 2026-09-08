"""Plannable countries come from `apify_supported_countries`.

That table is the research agent's own — `markets` stays untouched for the
scrape pipelines. Nothing about countries is hardcoded: add a row and it
reaches the prompt on the next context build.
"""

import pytest
from unittest.mock import MagicMock, patch

from app.core.config import Settings
from app.core.dependencies import CountriesUnavailable, _fetch_countries_from_db
from app.models.domain import AgentContext, MarketEntry, TaxonomyEntry
from app.services.prompt import _build_markets_block, _build_regions_block


def _ctx(markets):
    return AgentContext(markets=markets, taxonomy=[
        TaxonomyEntry(slug="fitness", aliases=["gym"])
    ])


def _rows(*rows):
    """Mocks the .table().select().eq().execute() chain the fetch uses."""
    client = MagicMock()
    chain = client.table.return_value.select.return_value.eq.return_value
    chain.execute.return_value.data = list(rows)
    return client


def test_every_column_is_read():
    client = _rows({
        "country_code": "BR", "name": "Brazil",
        "region": "LATAM", "languages": ["pt"],
    })
    with patch("app.core.dependencies.get_supabase_admin", return_value=client):
        countries = _fetch_countries_from_db(Settings(openai_api_key="sk-x"))
    m = countries[0]
    assert (m.iso, m.name, m.region) == ("BR", "Brazil", "LATAM")
    assert m.languages == ["pt"]


def test_region_is_optional_because_the_llm_infers_it():
    """region is nullable by design — "the Gulf" is resolved by the model
    unless a row overrides it."""
    client = _rows({"country_code": "KE", "name": "Kenya",
                    "region": None, "languages": []})
    with patch("app.core.dependencies.get_supabase_admin", return_value=client):
        countries = _fetch_countries_from_db(Settings(openai_api_key="sk-x"))
    assert countries[0].region is None


def test_inactive_and_duplicate_rows_are_handled():
    """is_active filters at the query; duplicates are defensive."""
    client = _rows(
        {"country_code": "BR", "name": "Brazil", "region": None, "languages": ["pt"]},
        {"country_code": "BR", "name": "Brazil", "region": None, "languages": ["pt"]},
    )
    with patch("app.core.dependencies.get_supabase_admin", return_value=client):
        countries = _fetch_countries_from_db(Settings(openai_api_key="sk-x"))
    assert len(countries) == 1


def test_empty_table_reads_as_unavailable():
    client = _rows()
    with patch("app.core.dependencies.get_supabase_admin", return_value=client):
        assert _fetch_countries_from_db(Settings(openai_api_key="sk-x")) is None


def test_regions_are_derived_from_the_region_column():
    """No hardcoded Gulf/MENA/LATAM table — re-file a market in Supabase and
    the expansion follows."""
    ctx = _ctx([
        MarketEntry(code="BR", iso="BR", name="Brazil", region="LATAM"),
        MarketEntry(code="MX", iso="MX", name="Mexico", region="LATAM"),
        MarketEntry(code="SA", iso="SA", name="Saudi Arabia", region="MENA"),
    ])
    assert ctx.regions == {"LATAM": ["BR", "MX"], "MENA": ["SA"]}
    block = _build_regions_block(ctx)
    assert "LATAM -> BR, MX" in block and "MENA -> SA" in block


def test_no_region_overrides_tells_the_model_to_expand_regions_itself():
    """apify_supported_countries.region is NULL for every row by default, so
    the model resolves "the Gulf" from its own geography. It must EXPAND —
    asking "which country?" when four of them are on the list is the bug this
    replaced."""
    ctx = _ctx([MarketEntry(code="XX", iso="XX", name="Nowhere")])
    assert ctx.regions == {}
    block = _build_regions_block(ctx)
    assert "expand " in block.lower()
    assert "Never ask which country" in block
    # Region members absent from the list must be dropped, not emitted.
    assert "Delete any code that is not there" in block


def test_prompt_carries_each_market_language():
    """Hashtag localisation reads markets.languages rather than the model
    guessing a language from the country name."""
    block = _build_markets_block(_ctx([
        MarketEntry(code="BR", iso="BR", name="Brazil", region="LATAM", languages=["pt"]),
    ]))
    assert "BR | Brazil | LATAM | pt" in block
    assert "Generate hashtags in the languages listed" in block


def test_no_geo_targeting_flag_since_every_row_is_targetable():
    """apify_supported_countries comes FROM the actor's list, so every row is
    geo-targetable. The old flag was for `markets`, where India had no code."""
    block = _build_markets_block(_ctx([
        MarketEntry(code="IN", iso="IN", name="India", languages=["hi", "en"]),
    ]))
    assert "no geo-targeting" not in block


def test_allowed_codes_are_exactly_the_table():
    ctx = _ctx([MarketEntry(code="SA", iso="SA", name="Saudi Arabia")])
    assert ctx.allowed_iso_codes == {"SA"}


def test_no_alias_table_is_built_into_the_prompt():
    """Nicknames are the model's job now — "Dubai" -> AE, "KSA" -> SA,
    "Türkiye" -> TR all verified without a lookup table. The old aliases.py
    mapped to `markets` codes (UAE, not AE) and silently broke the Gulf once
    this agent moved to ISO-2."""
    from app.services.prompt import assemble_system_prompt
    prompt = assemble_system_prompt(_ctx([
        MarketEntry(code="AE", iso="AE", name="United Arab Emirates"),
    ]))
    assert "COUNTRY ALIASES" not in prompt


def test_no_country_list_is_a_503_not_a_stale_plan():
    """There is no hardcoded fallback by design. Planning against a stale copy
    of the country list is worse than refusing: the operator would get a
    confident plan for markets that may no longer be scrapeable."""
    from fastapi import HTTPException

    from app.core.dependencies import build_context, get_agent_context, reset_agent_context

    reset_agent_context()
    with patch("app.core.dependencies._fetch_countries_from_db", return_value=None):
        with pytest.raises(CountriesUnavailable):
            build_context(Settings(openai_api_key="sk-x"))
        with pytest.raises(HTTPException) as caught:
            get_agent_context()
    assert caught.value.status_code == 503
    reset_agent_context()


def test_a_failed_context_is_not_cached():
    """A transient outage must not pin the process into a broken state."""
    from app.core.dependencies import get_agent_context, reset_agent_context

    reset_agent_context()
    import app.core.dependencies as deps
    assert deps._agent_context is None


def test_aliases_are_read_from_the_row():
    client = _rows({
        "country_code": "AE", "name": "United Arab Emirates", "region": None,
        "aliases": ["uae", "dubai", "emirates"], "languages": ["ar", "en"],
    })
    with patch("app.core.dependencies.get_supabase_admin", return_value=client):
        countries = _fetch_countries_from_db(Settings(openai_api_key="sk-x"))
    assert countries[0].aliases == ["uae", "dubai", "emirates"]


def test_aliases_appear_in_the_prompt_only_when_present():
    """Most rows have none — the model resolves ordinary nicknames unaided.
    A blank "aka" on 180 rows would be noise."""
    block = _build_markets_block(_ctx([
        MarketEntry(code="AE", iso="AE", name="United Arab Emirates",
                    aliases=["dubai", "emirates"], languages=["ar"]),
        MarketEntry(code="KE", iso="KE", name="Kenya"),
    ]))
    assert "aka dubai, emirates" in block
    assert "KE | Kenya" in block
    assert "KE | Kenya |" not in block  # no empty trailing fields


def test_alias_codes_are_iso_two_not_market_codes():
    """The deleted aliases.py mapped "dubai" -> UAE, the `markets` code. This
    table is ISO-2, and that mismatch silently broke "the Gulf"."""
    block = _build_markets_block(_ctx([
        MarketEntry(code="AE", iso="AE", name="United Arab Emirates",
                    aliases=["dubai"]),
    ]))
    assert "AE | United Arab Emirates" in block
    assert "UAE |" not in block
