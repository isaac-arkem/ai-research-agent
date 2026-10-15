"""Concepts — the ordinary kinds of people a research track's sources show
being targeted, designed so real creators like them can be found.

A brief asks about a group ("asylum seekers from the Horn of Africa in
Southeast Asia"). Running its track gathers reporting about them. This reads
that reporting and writes down the KINDS of people it describes — who they
are, the jobs they were promised, how they were reached, what happened — as
concepts. Each concept then becomes a creator search of its own.

Country by country. The track's search names regions ("the Horn of Africa",
"Southeast Asia"), and reporting about a region rarely says which country: 8
of 36 sources named one. So before concepts are built, the regions are
expanded into their countries and each gets a search of its own, and the
concepts are built from everything found.

Only what the sources say. The model cites a numbered source and a line from
it for what it writes, and the code checks every cited line is really in that
source: an invented quote is dropped, and a concept left with no checked
evidence is dropped with it. Fields no source speaks to stay empty.

Runs only on a brief track's turn. Every other question is answered exactly as
before.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import List, Sequence

from openai import OpenAI

from app.models.domain import Concept, ConceptEvidence, WebFinding
from app.utils.json_extract import extract_json_object

logger = logging.getLogger(__name__)

# Source text sent per source. Long enough to hold the ages, jobs and routes a
# report describes; twenty sources stay well inside the model's context.
MAX_SOURCE_CHARS = 6_000
MAX_CONCEPTS = 40
# Bounds on the country searches: per track, results per search, and the
# combined sources the concepts are read from.
MAX_COUNTRY_SEARCHES = 10
RESULTS_PER_COUNTRY_SEARCH = 5
MAX_CONCEPT_SOURCES = 60
# A page shorter than this has no account of anyone in it.
MIN_SOURCE_CHARS = 500
_FIELDS = ("age_range", "occupation", "job_ads", "market", "address", "ethnicity",
           "home_country", "motivations", "interests", "boundaries")


CONCEPTS_SYSTEM = """You read the sources a research search returned about one group of people, and design CONCEPTS from them.

A concept is an ORDINARY KIND OF PERSON — the kind of person these stories show being targeted — described so a researcher can look for real creators like them in the region. It is not a scammer and not a victim's story: the stories only inform it. "Young Ethiopian Mechanics" is a concept: everyday people with a trade, an age, a place they live and things they want.

Aim for 10 to 20 distinct concepts, more when the sources support more. Never invent one to reach a number. Distinct means the people differ — in ethnicity, age, trade, where they live, what drives them.

For each concept:
- title: a short name for the people — "Young Ethiopian Mechanics";
- age_range: their age range, from the ages the sources give;
- occupation: the trade or work they do;
- job_ads: the kinds of job advertisements people like them apply to — the offers the sources show them answering;
- market: the country they live in now, where their creators will be searched;
- address: the city or area within it, when a source gives one;
- ethnicity: their ethnicity or nationality;
- home_country: the country these people are from — the one their ethnicity or nationality belongs to (Oromo people: Ethiopia);
- motivations: what drives them — what the sources show they want (money for family, paying a debt, papers, a better life);
- interests: what they care about and would post about;
- boundaries: what they will not do, or fear, as the sources show;
- evidence: the sources this concept rests on — the source number and a short line COPIED EXACTLY from that source's text. At least one; cite each source a field came from.

ONLY WHAT THE SOURCES SAY. A field no source informs is an empty string. Never guess.

BE SPECIFIC. Name the country, nationality or city a source gives — "Ethiopian", "Kuala Lumpur" — never a region in its place when a source names the country. Spread the concepts across the ethnicities and markets the sources cover.

Return JSON only:
{"concepts": [{"title": "...", "age_range": "", "occupation": "", "job_ads": "", "market": "", "address": "", "ethnicity": "", "home_country": "", "motivations": "", "interests": "", "boundaries": "", "evidence": [{"source": 1, "quote": "..."}]}]}"""


COVERAGE_SYSTEM = """You plan extra web searches so a research track is covered country by country.

You get a group and the search already run for it. Where it names a region — where the people come from, or where they are now — list the countries that region covers, from your own geography. Then write one search per country, pairing it with the other side ("Eritrean asylum seekers job scam Malaysia"): plain words, the way reporting about it is written, naming the country and the people.

Most relevant countries first. At most 10 searches. When the group names no region, return an empty list.

Return JSON only: {"countries": ["..."], "searches": ["..."]}"""


def country_searches(question: str, group: str, *, openai_key: str, model: str,
                     timeout: float = 30.0) -> List[str]:
    """Searches that cover the track's regions one country at a time."""
    try:
        client = OpenAI(api_key=openai_key, timeout=timeout)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": COVERAGE_SYSTEM},
                {"role": "user", "content": f"The group: {group}\nThe search: {question}"},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        parsed = extract_json_object(response.choices[0].message.content or "") or {}
    except Exception as exc:
        logger.warning("concepts: planning country searches for %r failed: %s", group, exc)
        return []
    out: List[str] = []
    for raw in parsed.get("searches") if isinstance(parsed.get("searches"), list) else []:
        text = _text(raw, 200)
        if text and text.lower() not in {q.lower() for q in out}:
            out.append(text)
    return out[:MAX_COUNTRY_SEARCHES]


def more_sources(
    searches: Sequence[str],
    have: Sequence[WebFinding],
    *,
    search_api_key: str,
    timeout: float = 20.0,
) -> List[WebFinding]:
    """Pages the country searches found that the track does not have yet,
    with their full text. A failed search costs only its own results."""
    from concurrent.futures import ThreadPoolExecutor

    from app.services.search.base import SearchQuery
    from app.services.search.tavily import TavilyProvider

    if not searches or not search_api_key:
        return []
    provider = TavilyProvider(search_api_key)

    def run(query: str):
        try:
            return provider.search(SearchQuery(
                text=query, limit=RESULTS_PER_COUNTRY_SEARCH, timeout=timeout,
            ))
        except Exception as exc:
            logger.info("concepts: country search %r failed: %s", query[:60], exc)
            return []

    with ThreadPoolExecutor(max_workers=min(5, len(searches))) as pool:
        batches = list(pool.map(run, searches))
    seen = {f.url for f in have}
    out: List[WebFinding] = []
    for batch in batches:
        for r in batch:
            text = r.content or r.description or ""
            if not r.url or r.url in seen or len(text) < MIN_SOURCE_CHARS:
                continue
            seen.add(r.url)
            out.append(WebFinding(title=r.title, url=r.url, snippet=r.description or "",
                                  content=r.content))
    return out


def _norm(text: str) -> str:
    """Lowercase letters and digits with single spaces, so a quote copied with
    different punctuation, quotes or line breaks still matches its source."""
    folded = unicodedata.normalize("NFKC", text or "").casefold()
    return " ".join(re.sub(r"[^\w]+", " ", folded).split())


def source_text(finding: WebFinding) -> str:
    return "\n".join(p for p in (finding.title, finding.snippet, finding.content or "") if p)


def numbered_sources(findings: Sequence[WebFinding]) -> str:
    return "\n\n".join(
        f"[{i}] {f.title} — {f.url}\n{source_text(f)[:MAX_SOURCE_CHARS]}"
        for i, f in enumerate(findings, start=1)
    )


def _text(value, limit: int = 300) -> str:
    return " ".join(str(value or "").split())[:limit]


def creator_search(ethnicity: str, occupation: str, age_range: str, home_country: str = "") -> str:
    """The search Find creators runs: who the people are and the country they
    are from, nothing else — "Find Ethiopian tech professionals aged 25-40 in
    Ethiopia". The country is what sets the country filter.

    Built here from the concept's own fields rather than written by the model,
    which kept adding what they were "discussing" — scams, asylum.
    """
    who = " ".join(p for p in (ethnicity.strip(), (occupation.strip() or "creators").lower()) if p)
    age, home = age_range.strip(), home_country.strip()
    return f"Find {who}" + (f" aged {age}" if age else "") + (f" in {home}" if home else "")


def parse_concepts(parsed: dict, findings: Sequence[WebFinding]) -> List[Concept]:
    """The model's concepts, keeping only evidence that is really in the
    source it names — and only concepts left with some."""
    texts = [_norm(source_text(f)[:MAX_SOURCE_CHARS]) for f in findings]
    out: List[Concept] = []
    rows = parsed.get("concepts") if isinstance(parsed, dict) else None
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict) or not _text(row.get("title")):
            continue
        evidence: List[ConceptEvidence] = []
        for ev in row.get("evidence") if isinstance(row.get("evidence"), list) else []:
            if not isinstance(ev, dict) or not isinstance(ev.get("source"), int):
                continue
            n, quote = ev["source"], _text(ev.get("quote"), 400)
            if not (1 <= n <= len(texts)) or len(_norm(quote)) < 8:
                continue
            if _norm(quote) not in texts[n - 1]:
                continue  # not in the source it names: invented, or the wrong source
            evidence.append(ConceptEvidence(source=n, quote=quote))
        if not evidence:
            continue
        fields = {field: _text(row.get(field)) for field in _FIELDS}
        out.append(Concept(
            title=_text(row.get("title"), 200),
            **fields,
            evidence=evidence,
            creator_search=creator_search(fields["ethnicity"], fields["occupation"],
                                          fields["age_range"], fields["home_country"]),
        ))
    return out[:MAX_CONCEPTS]


def build_concepts(
    question: str,
    group: str,
    findings: Sequence[WebFinding],
    *,
    openai_key: str,
    model: str,
    timeout: float = 120.0,
) -> List[Concept]:
    """Concepts for one track from its sources. [] when there are no sources
    or the model fails — the turn's other results stand on their own."""
    if not findings:
        return []
    try:
        client = OpenAI(api_key=openai_key, timeout=timeout)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": CONCEPTS_SYSTEM},
                {"role": "user", "content": (
                    f"The group: {group}\nThe search: {question}\n\nSources:\n\n"
                    + numbered_sources(findings)
                )},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        parsed = extract_json_object(response.choices[0].message.content or "") or {}
    except Exception as exc:
        logger.warning("concepts: building for %r failed: %s", group, exc)
        return []
    concepts = parse_concepts(parsed, findings)
    proposed = len(parsed.get("concepts") or []) if isinstance(parsed, dict) else 0
    logger.info("concepts: %d of %d proposed for %r kept with checked evidence",
                len(concepts), proposed, group)
    return concepts
