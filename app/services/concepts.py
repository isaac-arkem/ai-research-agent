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

ONLY THE GROUP'S PEOPLE. Every concept is people who belong to the group you are given — from the place the group is from. The sources also describe others; leave them out. A concept's title names ordinary people by who they are and what they do — "Young Ethiopian Mechanics" — never "victims", "scam" or what happened to them.

Aim for 10 to 20 distinct concepts, more when the sources support more. Never invent one to reach a number. Distinct means the people differ — in ethnicity, age, trade, where they live, what drives them.

For each concept:
- title: a short name for the people — "Young Ethiopian Mechanics";
- age_range: their age range, from the ages the sources give;
- occupation: the people, named by their trade — a plural noun for the people who do the work ("cleaners", "construction labourers", "drivers", "tech professionals"), never a category of work or a situation ("menial jobs", "job opportunities", "unemployed");
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
    """The search Find creators runs: PEOPLE, worded the way pages that list
    creators are written — "Find Ethiopian content creators and influencers in
    Ethiopia who are tech workers, aged 25-30".

    Worded as a labour question ("…living in Ethiopia (occupation: tech
    workers)"), the web search returned job boards and workforce reports;
    worded like this it returned "Top 20 Ethiopian Influencers" and "Tech
    Influencer in Ethiopia". Creators is always the subject — never the job.
    Built here from the concept's fields, not written by the model.
    """
    who = " ".join(p for p in (ethnicity.strip(), "content creators and influencers") if p)
    age, home, job = age_range.strip(), home_country.strip(), occupation.strip()
    return (
        f"Find {who}"
        + (f" in {home}" if home else "")
        + (f" who are {job.lower()}" if job else "")
        + (f", aged {age}" if age else "")
    )


TRADE_CHECK_SYSTEM = """For each numbered entry, say whether it names people by a trade or kind of work they do — "cleaners", "tech workers", "drivers" are; "asylum seekers", "varied", "job seekers", "menial jobs", "unemployed" are not.

Return JSON only: {"trades": [true, false, ...]} — one answer per entry, in order."""


GROUP_COUNTRIES_SYSTEM = """You are given a group of people from a research brief. List the countries the people of this group are FROM — their origin, not where they are now.

Read a region in its strict, narrow sense — the core countries the name means — never a wider or extended grouping that some organisations use. When the group names no place of origin, return an empty list.

Return JSON only: {"countries": ["..."]}"""


def group_countries(group: str, *, openai_key: str, model: str, timeout: float = 30.0) -> List[str]:
    """The countries a track's group comes from, [] when it names none or the
    call fails — then no concept is dropped for its country."""
    try:
        client = OpenAI(api_key=openai_key, timeout=timeout)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": GROUP_COUNTRIES_SYSTEM},
                {"role": "user", "content": f"The group: {group}"},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        parsed = extract_json_object(response.choices[0].message.content or "") or {}
    except Exception as exc:
        logger.warning("concepts: countries for %r failed: %s", group, exc)
        return []
    raw = parsed.get("countries") if isinstance(parsed.get("countries"), list) else []
    out: List[str] = []
    for value in raw:
        text = _text(value, 80)
        if text and text.casefold() not in {c.casefold() for c in out}:
            out.append(text)
    return out


def from_the_group(concepts: List[Concept], countries: Sequence[str]) -> List[Concept]:
    """Only the concepts whose people come from the group's countries.

    A code check, not an instruction: told "only the group's people", the
    model still wrote a Kenyan concept into a Horn of Africa track. A concept
    with no From country is kept — nothing shows it is outside the group.
    """
    if not countries:
        return concepts
    allowed = {_norm(c) for c in countries}
    kept, dropped = [], []
    for concept in concepts:
        home = _norm(concept.home_country)
        if not home or home in allowed or any(a and a in home for a in allowed):
            kept.append(concept)
        else:
            dropped.append(concept)
    if dropped:
        logger.info("concepts: dropped %d from outside %s: %s", len(dropped), list(countries),
                    [f"{c.title} ({c.home_country})" for c in dropped])
    return kept


def keep_only_trades(concepts: List[Concept], *, openai_key: str, model: str,
                     timeout: float = 30.0) -> List[Concept]:
    """Clear a concept's occupation when it is not a trade, so Find creators
    searches the people alone ("Find Somali creators aged 30-35 living in
    Somalia") instead of a situation ("occupation: asylum seekers").

    One call for every concept. On any failure the concepts are kept as they
    are — the search is still about people, only less exact.
    """
    asked = [i for i, c in enumerate(concepts) if c.occupation.strip()]
    if not asked:
        return concepts
    try:
        client = OpenAI(api_key=openai_key, timeout=timeout)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": TRADE_CHECK_SYSTEM},
                {"role": "user", "content": "\n".join(
                    f"{n}. {concepts[i].occupation}" for n, i in enumerate(asked, start=1)
                )},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        answers = (extract_json_object(response.choices[0].message.content or "") or {}).get("trades")
    except Exception as exc:
        logger.warning("concepts: trade check failed, occupations kept: %s", exc)
        return concepts
    if not isinstance(answers, list) or len(answers) != len(asked):
        logger.warning("concepts: trade check gave %r for %d entries, occupations kept", answers, len(asked))
        return concepts
    out = list(concepts)
    for i, is_trade in zip(asked, answers):
        if is_trade is False:
            c = out[i]
            out[i] = c.model_copy(update={
                "occupation": "",
                "creator_search": creator_search(c.ethnicity, "", c.age_range, c.home_country),
            })
    return out


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
    check_model: str = "",
) -> List[Concept]:
    """Concepts for one track from its sources. [] when there are no sources
    or the model fails — the turn's other results stand on their own."""
    if not findings:
        return []
    homes = group_countries(group, openai_key=openai_key, model=check_model or model)
    try:
        client = OpenAI(api_key=openai_key, timeout=timeout)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": CONCEPTS_SYSTEM},
                {"role": "user", "content": (
                    f"The group: {group}\n"
                    + (f"The group's people are from: {', '.join(homes)}\n" if homes else "")
                    + f"The search: {question}\n\nSources:\n\n"
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
    concepts = keep_only_trades(
        from_the_group(parse_concepts(parsed, findings), homes),
        openai_key=openai_key, model=check_model or model,
    )
    proposed = len(parsed.get("concepts") or []) if isinstance(parsed, dict) else 0
    logger.info("concepts: %d of %d proposed for %r kept with checked evidence",
                len(concepts), proposed, group)
    return concepts
