"""Reports lane — ReliefWeb, the UN-run index of humanitarian reports.

UN agencies, IOM, Amnesty and NGOs publish their findings here: dated,
attributed, and usually more careful about method than news coverage. That is
the evidence a research brief cites when it compares groups.

ReliefWeb only answers callers with an approved `appname`, requested once at
https://apidoc.reliefweb.int/parameters#appname and set as RELIEFWEB_APPNAME.
Without it the lane says so instead of pretending it searched.

Offered to the planner once RELIEFWEB_APPNAME is set; the operator can also
ask for it, or leave it out, by name. Free.
"""

from __future__ import annotations

import logging
from typing import List

import httpx

logger = logging.getLogger(__name__)

ENDPOINT = "https://api.reliefweb.int/v2/reports"
MAX_RESULTS = 20


class ReliefWebNotConfigured(RuntimeError):
    """No approved appname — the lane cannot ask."""


def _body(query: str, from_date: str, to_date: str, limit: int) -> dict:
    return {
        "query": {"value": query},
        "filter": {
            "field": "date.created",
            "value": {
                "from": f"{from_date}T00:00:00+00:00",
                "to": f"{to_date}T23:59:59+00:00",
            },
        },
        "fields": {"include": ["title", "url_alias", "date.created", "source.name", "body"]},
        "sort": ["date.created:desc"],
        "limit": max(1, min(limit, MAX_RESULTS)),
    }


def parse_reports(payload: dict) -> List[dict]:
    """ReliefWeb's response as lane items."""
    items: List[dict] = []
    rows = payload.get("data") if isinstance(payload, dict) else None
    for i, row in enumerate(rows if isinstance(rows, list) else [], start=1):
        fields = row.get("fields") if isinstance(row, dict) else None
        if not isinstance(fields, dict):
            continue
        title = str(fields.get("title") or "").strip()
        url = str(fields.get("url_alias") or "").strip()
        if not url and row.get("id"):
            url = f"https://reliefweb.int/node/{row['id']}"
        if not title or not url:
            continue
        sources = [
            str(s.get("name") or "").strip()
            for s in (fields.get("source") or []) if isinstance(s, dict)
        ]
        publisher = ", ".join(s for s in sources if s)
        created = str((fields.get("date") or {}).get("created") or "")[:10] or None
        body = " ".join(str(fields.get("body") or "").split())
        items.append({
            "id": f"RW{i}",
            "title": title,
            "url": url,
            # The publisher, not reliefweb.int: "IOM" or "Amnesty International"
            # is what a reader needs to weigh the report.
            "source_domain": publisher or "reliefweb.int",
            "snippet": body[:400],
            "body": body,
            "date": created,
            "relevance": 0.8,
            "why_relevant": f"Report{' by ' + publisher if publisher else ''}",
        })
    return items


def search_reports(query: str, from_date: str, to_date: str, config: dict) -> List[dict]:
    """Humanitarian reports matching the query, published inside the window."""
    appname = (config.get("RELIEFWEB_APPNAME") or "").strip()
    if not appname:
        raise ReliefWebNotConfigured(
            "RELIEFWEB_APPNAME is not set — request one at "
            "https://apidoc.reliefweb.int/parameters#appname"
        )
    response = httpx.post(
        ENDPOINT,
        params={"appname": appname},
        json=_body(query, from_date, to_date, int(config.get("RELIEFWEB_MAX_RESULTS") or MAX_RESULTS)),
        timeout=float(config.get("RELIEFWEB_TIMEOUT") or 20.0),
    )
    if response.status_code != 200:
        raise RuntimeError(f"reliefweb returned HTTP {response.status_code}: {response.text[:200]}")
    items = parse_reports(response.json())
    logger.info("reports: %d report(s) for %r", len(items), query[:80])
    return items
