#!/usr/bin/env python
"""Top up `apify_supported_countries` from the scrape actor's own list.

The actor's `proxyCountryCode` enum is the authority on which countries can
be geo-targeted, and it changes when the actor is rebuilt. This reads it live
and fills the table, so the country list is never a copy kept in code.

Reports by default; writes only with --apply, after printing what it will do.
It creates no tables and alters no columns — schema is migrations
(sql/002, sql/003). This only adds and updates ROWS.

USAGE
-----
    python scripts/sync_apify_countries.py                    # dry run
    python scripts/sync_apify_countries.py --apply            # insert missing
    python scripts/sync_apify_countries.py --apply --deactivate-removed
                                          # also flag countries the actor dropped

Needs APIFY_TOKEN in .env, plus the Supabase vars the app already uses.

NOTE ON ALIASES AND REGION: inserted rows get neither. Both are optional —
the model resolves "Dubai" and "the Gulf" unaided — and both are yours to set
in Supabase. This script never overwrites them on a country that already exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.dependencies import COUNTRIES_TABLE  # noqa: E402
from app.core.supabase import get_supabase_admin  # noqa: E402


def fetch_actor_countries(actor: str, token: str) -> Dict[str, str]:
    """The actor's `proxyCountryCode` enum — the authority on geo targets."""
    url = f"https://api.apify.com/v2/acts/{actor}/builds/default"
    response = httpx.get(url, params={"token": token}, timeout=30.0)
    response.raise_for_status()

    schema = response.json()["data"]["inputSchema"]
    if isinstance(schema, str):
        schema = json.loads(schema)

    field = schema["properties"]["proxyCountryCode"]
    return {
        code: title
        for code, title in zip(field["enum"], field["enumTitles"])
        if code != "None"
    }


def fetch_existing() -> Dict[str, dict]:
    client = get_supabase_admin(get_settings())
    if client is None:
        raise SystemExit("Supabase is not configured — check your .env")
    rows = (
        client.table(COUNTRIES_TABLE)
        .select("country_code, name, is_active")
        .execute()
        .data
    )
    return {r["country_code"]: r for r in rows if r.get("country_code")}


def diff(
    existing: Dict[str, dict], supported: Dict[str, str]
) -> Tuple[Dict[str, str], List[dict], List[dict]]:
    """(to_insert, dropped_by_actor, reactivatable)."""
    to_insert = {c: n for c, n in supported.items() if c not in existing}
    dropped = [
        row
        for code, row in existing.items()
        if code not in supported and row.get("is_active")
    ]
    reactivatable = [
        row
        for code, row in existing.items()
        if code in supported and not row.get("is_active")
    ]
    return to_insert, dropped, reactivatable


def insert_countries(to_insert: Dict[str, str]) -> int:
    """Add rows. region/aliases/languages are left at their defaults — those
    are yours to fill in, and this never touches an existing row."""
    client = get_supabase_admin(get_settings())
    rows = [{"country_code": c, "name": n} for c, n in sorted(to_insert.items())]
    for start in range(0, len(rows), 100):
        client.table(COUNTRIES_TABLE).insert(rows[start : start + 100]).execute()
    return len(rows)


def set_active(codes: List[str], active: bool) -> int:
    client = get_supabase_admin(get_settings())
    for code in codes:
        (
            client.table(COUNTRIES_TABLE)
            .update({"is_active": active})
            .eq("country_code", code)
            .execute()
        )
    return len(codes)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", default=None, help="Apify actor id (~ form)")
    parser.add_argument(
        "--apply", action="store_true",
        help="Write to Supabase. Without it this is a dry run.",
    )
    parser.add_argument(
        "--deactivate-removed", action="store_true",
        help="With --apply, set is_active=false on countries the actor dropped. "
             "Off by default: a country going dark stops the agent planning "
             "for it, which is a judgement call.",
    )
    args = parser.parse_args()

    settings = get_settings()
    token: Optional[str] = settings.apify_token
    if not token:
        raise SystemExit("APIFY_TOKEN is not set — add it to .env")

    actor = args.actor or settings.apify_tiktok_actor
    supported = fetch_actor_countries(actor, token)
    existing = fetch_existing()
    to_insert, dropped, reactivatable = diff(existing, supported)

    print(f"actor            {actor}")
    print(f"actor countries  {len(supported)}")
    print(f"{COUNTRIES_TABLE:16} {len(existing)} row(s)")
    print()
    print(f"to insert        {len(to_insert)}")
    if to_insert:
        preview = ", ".join(f"{c} {n}" for c, n in sorted(to_insert.items())[:8])
        print(f"   {preview}{' ...' if len(to_insert) > 8 else ''}")
    print(f"dropped by actor {len(dropped)}")
    for row in dropped:
        print(f"   {row['country_code']:4} {row['name']}")
    print(f"re-activatable   {len(reactivatable)}")
    for row in reactivatable:
        print(f"   {row['country_code']:4} {row['name']}")

    if not (to_insert or dropped or reactivatable):
        print("\nin sync — nothing to do")
        return 0

    if not args.apply:
        print("\nDRY RUN — nothing written.")
        print("  --apply                       insert the countries above")
        print("  --apply --deactivate-removed  also flag the dropped ones")
        return 0

    if to_insert:
        print(f"\ninserted {insert_countries(to_insert)} country(ies)")
        print("region, aliases and languages left empty — set them in Supabase")

    if reactivatable:
        codes = [r["country_code"] for r in reactivatable]
        print(f"re-activated {set_active(codes, True)} country(ies)")

    if dropped:
        if args.deactivate_removed:
            codes = [r["country_code"] for r in dropped]
            print(f"deactivated {set_active(codes, False)} country(ies)")
        else:
            print(f"{len(dropped)} dropped country(ies) left active "
                  "(pass --deactivate-removed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
