# Fan out to the research lanes, fuse what comes back, rank it.
#
# This replaces the engine's own pipeline.py (5,578 lines), which we did not
# vendor. Almost all of that file is CLI concerns — flag parsing, terminal
# rendering, HTML publishing, the setup wizard, the save-to-disk library, the
# discovery protocol. The part that actually researches is the shape below:
#
#     plan -> lanes -> normalize -> annotate -> fuse -> rerank -> cluster
#
# Two invariants the engine depends on, both silent when broken:
#
#   * `streams` is keyed by (subquery_label, source). weighted_rrf looks the
#     label up in the plan to find the subquery's weight; a wrong key is a
#     KeyError, a missing one drops the stream with no error at all.
#   * `resolved_handles` must be @-stripped and lowercased. rerank compares
#     raw strings, so "@Handle" never matches and a creator's own posts stay
#     buried under the entity-miss penalty they were meant to escape.
#
# The date window is a parameter here, never a constant. The engine's
# defaults assume a 30-day brief; researchAgent asks landscape questions where
# a two-year-old "creators to follow" post is still the best evidence there
# is. See WINDOW_* below.

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.services.research.engine import (
    cluster,
    fusion,
    grounding as web,
    hackernews,
    instagram,
    normalize,
    planner,
    polymarket,
    reddit,
    rerank,
    schema,
    signals,
    tiktok,
)

logger = logging.getLogger(__name__)

# NOTE ON NAMES: `engine.grounding` is the engine's WEB SEARCH lane and has
# nothing to do with app/services/grounding.py, which routes a conversation
# turn. Imported as `web` here so the two never read as the same thing.

# How far back a run looks when the caller does not say. Wide on purpose: a
# creator-landscape question is not a news question, and the ranker already
# prefers recent items on its own (rerank applies a 1/sqrt(age) weight
# regardless of window). Narrowing here would discard evidence twice.
DEFAULT_WINDOW_DAYS = 365

# Candidates carried out of fusion into reranking. The shortlist below is what
# the LLM actually reads; the rest are scored by the local fallback.
DEFAULT_POOL_LIMIT = 60
DEFAULT_SHORTLIST = 25


@dataclass
class LaneOutcome:
    """What one lane did. Distinguishes the three states that matter.

    "ok with 0 items" and "raised" are different facts about the world — the
    first says the platform has nothing, the second says we failed to ask.
    Collapsing them is how a dead credential reads as an empty market.
    """

    source: str
    state: str  # "ok" | "no-results" | "skipped" | "error"
    items: int = 0
    detail: str = ""


@dataclass
class ResearchResult:
    topic: str
    window: Tuple[str, str]
    candidates: List[schema.Candidate] = field(default_factory=list)
    clusters: List[schema.Cluster] = field(default_factory=list)
    source_status: Dict[str, str] = field(default_factory=dict)
    lane_outcomes: List[LaneOutcome] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        """Whether any lane actually returned evidence."""
        return any(o.state == "ok" and o.items for o in self.lane_outcomes)


def window_for(
    days: int = DEFAULT_WINDOW_DAYS,
    *,
    as_of: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> Tuple[str, str]:
    """Resolve a (from, to) window.

    Explicit dates win; otherwise `days` back from `as_of` (default today).
    Callers that want no practical restriction pass a large `days` rather than
    a sentinel — every downstream filter compares dates, so a real date keeps
    that machinery honest instead of teaching it about a None case.
    """
    if from_date and to_date:
        return from_date, to_date
    end = date.fromisoformat(as_of) if as_of else date.today()
    return (end - timedelta(days=max(1, days))).isoformat(), end.isoformat()


# ---------------------------------------------------------------------------
# Lanes
# ---------------------------------------------------------------------------
#
# Each returns raw item dicts in its own source's shape. normalize handles the
# differences; nothing here should know what a TikTok item looks like.


def _lane_reddit(q: str, fd: str, td: str, depth: str, cfg: dict, opts: dict) -> List[dict]:
    result = reddit.search_and_enrich(
        q, fd, td, depth=depth, token="", subreddits=opts.get("subreddits")
    )
    return reddit.parse_reddit_response(result)


def _lane_hackernews(q: str, fd: str, td: str, depth: str, cfg: dict, opts: dict) -> List[dict]:
    result = hackernews.search_hackernews(q, fd, td, depth=depth)
    return hackernews.parse_hackernews_response(result, query=q)


def _lane_tiktok(q: str, fd: str, td: str, depth: str, cfg: dict, opts: dict) -> List[dict]:
    # token="" forces the Apify branch inside search_and_enrich. We have no
    # ScrapeCreators account, so this is the only path that can answer.
    result = tiktok.search_and_enrich(
        q, fd, td, depth=depth, token="",
        # Unlike Instagram, this lane takes hashtags directly alongside the
        # keyword query, so both signals are used.
        hashtags=opts.get("tiktok_hashtags") or opts.get("hashtags"),
        creators=opts.get("tiktok_creators"),
        apify_token=cfg.get("APIFY_API_TOKEN"),
    )
    return tiktok.parse_tiktok_response(result)


def _lane_instagram(q: str, fd: str, td: str, depth: str, cfg: dict, opts: dict) -> List[dict]:
    # search_instagram_apify takes no hashtags argument — it derives one from
    # the topic via _to_hashtag_form(_extract_core_subject(topic)). Handed the
    # planner's keyword query that yields #modestfashioncreatorssaudiarabia,
    # which matches nothing. So a resolved hashtag IS the query here.
    # Every resolved hashtag reaches the actor, which takes a list — so three
    # real tags cost the same single run as one. Taking hashtags[0] threw the
    # rest away, and the resolver orders them most specific first, which is
    # the one likeliest to have no posts behind it.
    hashtags = opts.get("hashtags") or []
    result = instagram.search_and_enrich(
        hashtags[0] if hashtags else q, fd, td, depth=depth, token="",
        ig_creators=opts.get("ig_creators"),
        apify_token=cfg.get("APIFY_API_TOKEN"),
        hashtags=hashtags,
    )
    return instagram.parse_instagram_response(result)


def _lane_polymarket(q: str, fd: str, td: str, depth: str, cfg: dict, opts: dict) -> List[dict]:
    result = polymarket.search_polymarket(q, fd, td, depth=depth)
    return polymarket.parse_polymarket_response(result, topic=q)


def _lane_web(q: str, fd: str, td: str, depth: str, cfg: dict, opts: dict) -> List[dict]:
    """The web lane — Tavily, asked the operator's own question.

    `web_search` picks the backend from config; with TAVILY_API_KEY set and no
    Brave/Exa/Serper key it resolves to tavily. Passing the market through
    `LAST30DAYS_COUNTRY` is what makes this the only geo-targeted lane in the
    engine, and it only applies on the general topic (see tavily_search).
    """
    config = dict(cfg)
    if opts.get("country_name"):
        config["LAST30DAYS_COUNTRY"] = opts["country_name"]
    if opts.get("tavily_topic"):
        config["TAVILY_TOPIC"] = opts["tavily_topic"]

    # The topic as the router settled it, NOT the planner's keyword rewrite.
    #
    # planner._build_prompt tells the model to write something "concise and
    # keyword-heavy" that "matches how content is TITLED on platforms". That
    # is right for Reddit, Hacker News and TikTok, which match titles. Tavily
    # does its own query understanding and rewards a natural question — and
    # app/services/grounding.py records the experiment: the same question
    # typed into Tavily's dashboard returned more handles than our rewrite of
    # it, because "the rewrite drops the words carrying the intent".
    #
    # Vendoring the engine silently reintroduced that paraphrase. Observed:
    # "list popular music artist in Ghana in 2026" reached Tavily as "top
    # Instagram influencers Ghana" — the subject gone, the results a mixture.
    #
    # Follow-ups are unaffected. The router has already rewritten a dependent
    # message into a standalone topic ("what about Ghana, senegal" becomes
    # "dark skinned influencers in Ghana and Senegal"), which is exactly the
    # natural-language string this lane wants.
    query = opts.get("raw_topic") or q
    items, _artifact = web.web_search(query, (fd, td), config, backend=opts.get("web_backend", "auto"))
    return items


# The engine calls the web source "grounding"; the plan's source names have to
# match, so the key does too. `web` is the alias a caller is likely to write.
LANES: Dict[str, Callable[..., List[dict]]] = {
    "reddit": _lane_reddit,
    "hackernews": _lane_hackernews,
    "tiktok": _lane_tiktok,
    "instagram": _lane_instagram,
    "polymarket": _lane_polymarket,
    "grounding": _lane_web,
    "web": _lane_web,
}

# What the planner is allowed to route to. Anything absent here it will never
# put in a subquery, which is how a source we cannot serve stays out of a plan
# instead of failing at fan-out time.
AVAILABLE_SOURCES = ["grounding", "reddit", "hackernews", "instagram", "tiktok", "polymarket"]

# Which lanes can answer each kind of question.
#
# A market question ("in what country can I get dark-skinned influencers")
# cannot be answered by Instagram or TikTok. Those lanes return POSTS — a
# single creator's reel, captioned #MelaninPoppin — and no post names the
# country you should enter or compares it to another. Left in the pool they
# do worse than fail: measured on that exact question they filled all five
# surviving findings and pushed every market report out of the ranking, so
# the run cost two Apify calls and returned one market instead of seven.
#
# What answers "which country" is a market report, a ranking, a per-country
# breakdown — the web lane, with discussion as a secondary read.
SOURCES_FOR_ANSWER = {
    "markets": ["grounding", "reddit", "hackernews"],
}

# Lanes that cost money per call. Everything else is keyless or flat-rate.
PAID_LANES = frozenset({"instagram", "tiktok"})


def paid_lane_allowed(source: str, named_platforms) -> bool:
    """A paid lane runs ONLY when the operator named that platform.

    Scraping is not free and it is usually not the answer. The web lane
    already reaches the directories, awards lists and rankings that name
    people; the social lanes add handles and engagement for a platform someone
    has actually chosen.

    Running them on every creator question billed Apify to answer what the web
    lane had already answered — and worse, flooded the pool: TikTok returned
    96 items against the web lane's 5 on "Top Ghanaian Music Artists", pushing
    the pages that named the artists out of the results entirely.

    So the trigger is explicit mention, nothing else. No weight threshold, no
    shape-based forcing. "Top Ghanaian Music Artists" is answered from the
    web; "top Ghanaian artists on TikTok" scrapes TikTok.
    """
    if source not in PAID_LANES:
        return True
    return source in {str(p).strip().lower() for p in (named_platforms or ())}


def plan_for(
    topic: str,
    *,
    provider=None,
    model: Optional[str] = None,
    depth: str = "default",
    sources: Optional[List[str]] = None,
    context: str = "",
    answer: Optional[str] = None,
) -> schema.QueryPlan:
    """Let the engine decide which sources answer this question.

    This is the routing brain: `plan_query` asks the LLM for `source_weights`
    and a per-subquery `sources` list, then `_sanitize_plan` normalizes the
    answer and fills what the model left out. With no provider it falls back
    to a deterministic plan keyed off the inferred intent — weaker, but it
    still routes (prediction leans polymarket, how_to leans youtube, and so
    on; see planner._default_source_weights).

    The prompt behind it carries rules earned from real failures — strip
    temporal phrases from search_query, never emit meta-research phrases,
    paraphrase intent-modifiers into several subqueries rather than echoing
    the literal words. Worth reading before overriding any of it.
    """
    # An explicit `sources` wins; otherwise the answer shape narrows the pool
    # to the lanes that can actually answer it.
    available = list(sources or SOURCES_FOR_ANSWER.get(answer or "", AVAILABLE_SOURCES))
    if not (provider and model):
        return planner.plan_query(
            topic=topic, available_sources=available, requested_sources=None,
            depth=depth, provider=provider, model=model, context=context,
        )

    prompt = planner._build_prompt(topic, available, None, depth)
    if context:
        prompt += f"\n\nCurrent context (from web search): {context}"
    try:
        raw = provider.generate_json(model, prompt)
    except Exception as exc:
        logger.warning("planner LLM failed, using deterministic plan: %s", exc)
        return planner.plan_query(
            topic=topic, available_sources=available, requested_sources=None,
            depth=depth, provider=None, model=None, context=context,
        )

    raw = _rescale_source_weights(raw)
    plan = planner._sanitize_plan(raw, topic, available, None, depth)
    return _ensure_always_on(plan, available)


# Lanes that run on every plan regardless of what the planner chose.
#
# The planner picks per-subquery sources non-deterministically: measured across
# seven topics it left `grounding` out of every plan, then included it on a
# re-run of one of them. The weights it assigns are stable and sensible; which
# sources land in a given subquery is not.
#
# For a flat-rate lane that variance is pure loss. Tavily costs the same per
# search whether or not the planner remembered it, and it is the ONLY lane that
# geo-targets — so a creator question that happens to omit it loses the market-
# specific directory pages that produce most of the usable handles. Paid lanes
# are deliberately NOT here: for those the planner's judgement is what stands
# between a question and an Apify bill.
ALWAYS_ON_SOURCES = frozenset({"grounding"})


def _ensure_always_on(plan: schema.QueryPlan, available: List[str]) -> schema.QueryPlan:
    """Put the always-on lanes into every subquery that lacks them.

    Weights are left exactly as the planner set them, so this changes what gets
    RETRIEVED, never how it RANKS. A lane the planner thought unimportant still
    ranks low; it just is not silently absent.
    """
    forced = sorted(s for s in ALWAYS_ON_SOURCES if s in available)
    if not forced:
        return plan

    # A source absent from source_weights fuses at weighted_rrf's 1.0 default,
    # which would outrank every lane the planner actually chose. Give it the
    # lowest weight already in the plan: present, but never promoted past
    # something the planner preferred.
    floor = min(plan.source_weights.values(), default=1.0) if plan.source_weights else 1.0
    for source in forced:
        plan.source_weights.setdefault(source, floor)

    for subquery in plan.subqueries:
        for source in forced:
            if source not in subquery.sources:
                subquery.sources.append(source)
    return plan


TARGETING_PROMPT = """You resolve platform targeting for a social research run.

Topic: {topic}

Return JSON only, with this shape:
{{"hashtags": ["..."], "subreddits": ["..."], "ig_creators": [], "tiktok_creators": []}}

"hashtags" — 2-5 tags people ACTUALLY post under, without the "#". This is the
whole job. The Instagram and TikTok lanes search a hashtag literally, so a tag
has to be one real community uses, not a description of the topic compressed
into one word.

  topic "modest fashion creators Saudi Arabia"
    GOOD: ["modestfashion", "saudifashion", "abaya", "hijabstyle"]
    BAD:  ["modestfashioncreatorssaudiarabia"]   <- nobody posts under this

Prefer short established tags over long specific ones, and include the local
or market-specific tag when the topic names a place. A market rarely has its
own tag for a niche, so pair the niche tag with the market tag instead of
inventing a combined one.

"subreddits" — 2-5 subreddit names without "r/", where this is actually
discussed. Include the country's own subreddit when the topic names a market.

"ig_creators" / "tiktok_creators" — handles WITHOUT "@", only when the topic
names specific people or you are certain they exist. An invented handle costs
a paid scrape and returns nothing, so an empty list is the right answer
whenever you are guessing."""


def resolve_targets(
    topic: str,
    *,
    provider=None,
    model: Optional[str] = None,
) -> Dict[str, List[str]]:
    """Resolve hashtags, subreddits and handles for the social lanes.

    This is the engine's Step 0.55, which upstream expects the HOST agent to
    perform with its own web search and pass in as --tiktok-hashtags /
    --subreddits / --ig-creators. Nothing performs it automatically, and
    without it the social lanes are unusable: they take the planner's keyword
    query and compress it to a hashtag, so "modest fashion creators Saudi
    Arabia" becomes #modestfashioncreatorssaudiarabia and returns nothing.

    Empty lists on failure. Every consumer treats missing targeting as "no
    pre-resolved targets", which is the pre-existing behaviour.
    """
    empty: Dict[str, List[str]] = {
        "hashtags": [], "subreddits": [], "ig_creators": [], "tiktok_creators": [],
    }
    if not (provider and model):
        return empty
    try:
        raw = provider.generate_json(model, TARGETING_PROMPT.format(topic=topic))
    except Exception as exc:
        logger.warning("targeting resolution failed, lanes run untargeted: %s", exc)
        return empty

    def clean(key: str, strip: str) -> List[str]:
        values = raw.get(key)
        if not isinstance(values, list):
            return []
        out = []
        for value in values:
            if not isinstance(value, str):
                continue
            text = value.strip().lstrip(strip).strip()
            if text and text not in out:
                out.append(text)
        return out[:5]

    return {
        "hashtags": clean("hashtags", "#"),
        "subreddits": clean("subreddits", "#r/"),
        "ig_creators": clean("ig_creators", "@"),
        "tiktok_creators": clean("tiktok_creators", "@"),
    }


def _rescale_source_weights(raw: dict) -> dict:
    """Put the model's source weights on the scale _sanitize_plan expects.

    _sanitize_plan fills every source the model did NOT name with a base
    weight of 1.0, then normalizes the lot. That is correct only if the model
    also writes weights around 1.0. Ask a model for "source_weights" and it
    will just as readily return a distribution summing to 1.0 — and then the
    sources it deliberately left out come in at 1.0 each and swamp the ones it
    chose.

    Observed, for "find modest fashion creators in Saudi Arabia on Instagram":
    the model returned grounding 0.3, reddit 0.25, instagram 0.25, tiktok 0.2
    and named neither hackernews nor polymarket. After the fill those two
    ranked joint-first at 0.333 while grounding fell to 0.10.

    So rescale the named weights to a mean of 1.0 before handing them over.
    The ordering the model chose is preserved exactly, and an unnamed source
    now enters at the average of the named ones — which is what "base weight"
    was meant to mean.
    """
    weights = raw.get("source_weights")
    if not isinstance(weights, dict):
        return raw
    values = [
        float(v) for v in weights.values()
        if isinstance(v, (int, float)) and not isinstance(v, bool) and float(v) > 0
    ]
    if not values:
        return raw
    mean = sum(values) / len(values)
    if mean <= 0:
        return raw
    rescaled = {
        source: (float(value) / mean)
        for source, value in weights.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    out = dict(raw)
    out["source_weights"] = rescaled
    return out


def _run_lane(
    source: str, subquery: schema.SubQuery, fd: str, td: str,
    depth: str, cfg: dict, opts: dict,
) -> Tuple[str, schema.SubQuery, List[dict], LaneOutcome]:
    lane = LANES.get(source)
    if lane is None:
        return source, subquery, [], LaneOutcome(source, "error", 0, "no lane registered")
    try:
        items = lane(subquery.search_query, fd, td, depth, cfg, opts) or []
    except Exception as exc:
        # One lane dying must not take the run with it. A landscape brief from
        # four sources is worth having; the outcome records what was lost.
        logger.warning("lane %s failed: %s", source, exc)
        return source, subquery, [], LaneOutcome(source, "error", 0, f"{type(exc).__name__}: {exc}")
    state = "ok" if items else "no-results"
    return source, subquery, items, LaneOutcome(source, state, len(items))


def run_research(
    *,
    topic: str,
    plan: schema.QueryPlan,
    config: dict,
    provider=None,
    model: Optional[str] = None,
    window: Optional[Tuple[str, str]] = None,
    depth: str = "default",
    pool_limit: int = DEFAULT_POOL_LIMIT,
    shortlist_size: int = DEFAULT_SHORTLIST,
    resolved_handles: Optional[set] = None,
    answer: Optional[str] = None,
    force_lanes: Optional[List[str]] = None,
    subjects: Optional[List[str]] = None,
    max_workers: int = 6,
    **lane_options: Any,
) -> ResearchResult:
    """Run one research pass and return ranked, clustered candidates.

    `provider`/`model` are the reasoning seam. Omit them and reranking falls
    back to the local heuristic — supported, but noticeably worse: the LLM
    score is 60% of a candidate's final rank.

    `lane_options` passes through per-lane targeting resolved upstream:
    subreddits, tiktok_hashtags, tiktok_creators, ig_creators.
    """
    from_date, to_date = window or window_for()
    window_days = max(1, (date.fromisoformat(to_date) - date.fromisoformat(from_date)).days)

    # Handles must be normalized before rerank compares them. Doing it here
    # rather than trusting the caller keeps the first-party credit working.
    handles = {h.lstrip("@").strip().lower() for h in (resolved_handles or set()) if h}

    streams: Dict[Tuple[str, str], List[schema.SourceItem]] = {}
    outcomes: List[LaneOutcome] = []

    jobs: List[Tuple[str, schema.SubQuery]] = []
    # The platforms the operator actually named. This is the whole scrape
    # policy: a paid lane runs when it is in here and not otherwise.
    named = [str(p).strip().lower() for p in (force_lanes or []) if p in LANES]
    for subquery in plan.subqueries:
        # A named platform the planner left out of this subquery still has to
        # be asked — the operator said it explicitly.
        for source in list(subquery.sources) + [
            p for p in named if p not in subquery.sources
        ]:
            # A question about ONE named person is not a hashtag sweep. The
            # paid lanes search a tag and return whoever posted under it,
            # which for "what are Sarkodie's handles" is fan pages, blogs and
            # update accounts — 35 of them, none of them him. His own handles
            # come off the web lane, which reads his profile pages.
            #
            # So the lane is not run and then filtered, it is not run. The
            # filtering came first and was the wrong half of the fix: it made
            # the answer clean while still paying Apify for every account it
            # threw away.
            if subjects and source in PAID_LANES:
                who = ", ".join(subjects)
                outcomes.append(LaneOutcome(
                    source, "skipped", 0,
                    f"not scraped — the question is about {who}, "
                    "and a hashtag sweep returns whoever posted under the tag",
                ))
                continue
            if not paid_lane_allowed(source, named):
                # Declined before it costs anything, and recorded so the brief
                # can say the lane was never asked rather than found nothing.
                outcomes.append(LaneOutcome(
                    source, "skipped", 0,
                    "not scraped — no platform named in the request",
                ))
                continue
            jobs.append((source, subquery))

    # The router's topic, for lanes that want the question rather than a
    # keyword rewrite of it.
    lane_options = dict(lane_options)
    lane_options.setdefault("raw_topic", getattr(plan, "raw_topic", "") or topic)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(_run_lane, source, subquery, from_date, to_date, depth, config, lane_options)
            for source, subquery in jobs
        ]
        for future in as_completed(futures):
            source, subquery, raw_items, outcome = future.result()
            outcomes.append(outcome)
            if not raw_items:
                continue

            items = normalize.normalize_source_items(
                source, raw_items, from_date, to_date, plan.freshness_mode
            )
            if not items:
                # Normalization dropped everything — most often the date gate
                # (it is strict for the web/grounding source). Not the same as
                # the lane returning nothing, so it gets its own state.
                outcome.state = "no-results"
                outcome.detail = "all items dropped in normalization"
                continue

            items = signals.annotate_stream(
                items,
                subquery.ranking_query,
                plan.freshness_mode,
                reference_date=to_date,
                # The engine defaults this to 30. Passing the real window is
                # what makes freshness relative to the caller's question
                # instead of to a brief we are not writing.
                max_days=window_days,
            )
            streams[(subquery.label, source)] = items

    # Collapse per-lane outcomes into one state per source: a source queried by
    # three subqueries is one source in the report, and any success outranks a
    # failure on a sibling subquery.
    status: Dict[str, str] = {}
    rank = {"ok": 0, "no-results": 1, "skipped": 2, "error": 3}
    for outcome in outcomes:
        current = status.get(outcome.source)
        if current is None or rank[outcome.state] < rank[current]:
            status[outcome.source] = outcome.state

    result = ResearchResult(
        topic=topic,
        window=(from_date, to_date),
        source_status=status,
        lane_outcomes=outcomes,
    )
    if not streams:
        return result

    candidates = fusion.weighted_rrf(
        streams,
        plan,
        pool_limit=pool_limit,
        range_from=from_date,
        range_to=to_date,
        first_party_handles=handles or None,
    )
    if not candidates:
        return result

    candidates = rerank.rerank_candidates(
        topic=topic,
        plan=plan,
        candidates=candidates,
        provider=provider,
        model=model,
        shortlist_size=shortlist_size,
        resolved_handles=handles,
    )

    result.candidates = candidates
    result.clusters = cluster.cluster_candidates(candidates, plan)
    return result
