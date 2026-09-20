"""What the agent can DO, as callable tools rather than fields on a form.

The router used to answer one multiple-choice question and Python did the
rest. That shape lost information the model had already worked out: asked for
"beauty creators like @janedoe but not like @johnsmith", it understood the
exclusion perfectly and wrote it into `topic`, the only free-text field on
the form — because `subjects` is a flat list with no way to say "not this
one". Every time something did not fit, a field was added: `answer`, then
`subjects`, then nearly `exclude`. Enumerating intents, one level up from
enumerating vocabulary.

So the model calls these instead. Each is a thin wrapper over machinery that
already exists and is unchanged: the lanes, the seed resolver, the creator
and market extractors.

One rule holds throughout: the model decides WHAT to do, and these decide
what it is allowed to COST. search_social refuses unless the operator named
that platform, and refuses again once the paid budget for the turn is spent.
A refusal comes back as a readable sentence, not an exception, so the model
can act on it — usually by asking the operator instead.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from app.models.domain import AgentContext, ChatTurn, Creator, Hashtag, MarketFinding, WebFinding

logger = logging.getLogger(__name__)

# How many findings a search hands back before the model has to ask for a
# specific page. Twenty full pages would flood the context on every step.
COMPACT_SNIPPET = 220


@dataclass
class ToolContext:
    """Everything the tools need, plus the budget they spend against."""

    settings: Any
    ctx: AgentContext
    prompt: str
    history: Optional[Sequence[ChatTurn]] = None
    on_progress: Optional[Callable[..., None]] = None

    # Gathered as the turn runs. The loop reads these when the model answers,
    # so a creator's handle always comes from an extractor that read a real
    # page — never from the model writing one into its reply.
    findings: List[WebFinding] = field(default_factory=list)
    creators: List[Creator] = field(default_factory=list)
    hashtags: List[Hashtag] = field(default_factory=list)
    markets: List[MarketFinding] = field(default_factory=list)
    lane_status: Dict[str, str] = field(default_factory=dict)

    paid_calls: int = 0
    searches: int = 0

    @property
    def max_paid(self) -> int:
        return int(getattr(self.settings, "tools_max_paid_calls", 2))

    @property
    def max_searches(self) -> int:
        return int(getattr(self.settings, "tools_max_searches", 4))


def _named_platforms(tc: ToolContext) -> List[str]:
    """Platforms the OPERATOR named, anywhere in the thread."""
    from app.services.grounding import platforms_named

    texts = [tc.prompt]
    for turn in tc.history or []:
        role = getattr(turn, "role", None) or (turn.get("role") if isinstance(turn, dict) else None)
        if role == "user":
            content = getattr(turn, "content", None) or (
                turn.get("content") if isinstance(turn, dict) else None
            )
            texts.append(str(content or ""))
    out: List[str] = []
    for text in texts:
        for platform in platforms_named(text):
            if platform not in out:
                out.append(platform)
    return out


# ── the tools ────────────────────────────────────────────────────────


def search_web(tc: ToolContext, query: str, country: Optional[str] = None) -> str:
    """Tavily, through the same lane the pipeline uses. Returns a compact list."""
    from app.services.grounding import _to_finding
    from app.services.search import SearchQuery, provider_from_settings

    if tc.searches >= tc.max_searches:
        return f"refused: already searched {tc.searches} times this turn. Answer with what you have."
    tc.searches += 1
    market = None
    if country:
        from app.services.grounding import _market_for

        market = _market_for(tc.ctx, country.upper()[:2])
    try:
        provider = provider_from_settings(tc.settings)
        results = provider.search(SearchQuery(
            text=query,
            country=market.iso.lower() if market else None,
            country_name=market.name if market else None,
            limit=int(getattr(tc.settings, "search_results_per_query", 20)),
            timeout=float(getattr(tc.settings, "search_timeout", 15.0)),
        ))
    except Exception as exc:
        logger.warning("tools: web search failed: %s", exc)
        return f"the web search failed: {type(exc).__name__}. Try a different query or answer without it."

    fresh = [_to_finding(r) for r in results or []]
    start = len(tc.findings)
    tc.findings.extend(fresh)
    tc.lane_status["web"] = "ok" if fresh else "no-results"
    if not fresh:
        return "no results for that query."
    lines = [
        f"[{start + i + 1}] {f.title} — {f.url}\n{(f.snippet or '')[:COMPACT_SNIPPET]}"
        for i, f in enumerate(fresh)
    ]
    return (
        f"{len(fresh)} results. Numbers are source ids; read_source(id) gives the full page.\n\n"
        + "\n\n".join(lines)
    )


def read_source(tc: ToolContext, source_id: int) -> str:
    """The full text of one finding, for when a snippet is not enough."""
    index = int(source_id) - 1
    if index < 0 or index >= len(tc.findings):
        return f"no source {source_id}. Ids run 1 to {len(tc.findings)}."
    found = tc.findings[index]
    body = (found.content or found.snippet or "").strip()
    return f"[{source_id}] {found.title} — {found.url}\n\n{body[:6000]}" if body else "that page had no readable text."


def search_social(tc: ToolContext, platform: str, query: str) -> str:
    """Instagram or TikTok, via Apify. THIS ONE COSTS MONEY, so it is gated.

    The gate is here rather than in the flow on purpose. The model decides
    what to do; this decides what it may spend. A refusal is phrased so the
    model can act on it — normally by asking the operator which platform.
    """
    from app.services.research import orchestrator, reasoning
    from app.services.research.engine import schema

    platform = (platform or "").strip().lower()
    if platform not in {"instagram", "tiktok"}:
        return f"no lane for {platform!r}. Only instagram and tiktok can be searched."

    named = _named_platforms(tc)
    if platform not in named:
        return (
            f"refused: the operator has not asked for {platform}. Scraping costs money, "
            "so it runs only on a platform they named. Ask them which platform to use."
        )
    if tc.paid_calls >= tc.max_paid:
        return (
            f"refused: the paid budget for this turn is spent ({tc.paid_calls} of "
            f"{tc.max_paid}). Answer with what you have."
        )

    tc.paid_calls += 1
    try:
        client = reasoning.build_reasoning_client(tc.settings)
        model = getattr(tc.settings, "research_plan_model", "gpt-4o")
        targets = orchestrator.resolve_targets(query, provider=client, model=model)
        sub = schema.SubQuery(
            label="tool", search_query=query, ranking_query=query,
            sources=[platform], weight=1.0,
        )
        plan = schema.QueryPlan(
            intent="find", freshness_mode="any", cluster_mode="none",
            raw_topic=query, subqueries=[sub], source_weights={platform: 1.0},
        )
        result = orchestrator.run_research(
            topic=query, plan=plan, config=orchestrator_config(tc),
            provider=client, model=model,
            window=orchestrator.window_for(
                int(getattr(tc.settings, "research_window_days", 365))
            ),
            depth=str(getattr(tc.settings, "research_depth", "default")),
            force_lanes=[platform],
            **targets,
        )
    except Exception as exc:
        logger.warning("tools: %s search failed: %s", platform, exc)
        return f"the {platform} search failed: {type(exc).__name__}."

    for source, state in (result.source_status or {}).items():
        tc.lane_status[source] = state

    from app.services.grounding import creators_from_post_authors

    found = creators_from_post_authors(result.candidates)
    known = {(c.handle or "").lower() for c in tc.creators}
    fresh = [c for c in found if (c.handle or "").lower() not in known]
    tc.creators.extend(fresh)
    if not fresh:
        return f"{platform}: nothing usable came back."
    lines = [f"@{c.handle} — {c.name or ''} — {c.why}" for c in fresh[:40]]
    return f"{len(fresh)} accounts from {platform}:\n" + "\n".join(lines)


def profile(tc: ToolContext, handle: str, platform: str) -> str:
    """What one named account actually posts.

    The gap that made the loop scrape blindly. Asked for "creators like
    @janedoe", it had no way to find out what @janedoe is like — only a
    hashtag sweep — so it swept "beauty" and returned fifty-nine strangers.

    Gated on the BUDGET but not on the operator naming a platform, unlike
    search_social. The distinction is what was named: sweeping a hashtag is a
    fishing trip on a platform they chose, while this reads an account they
    named themselves. Making them also name the platform would be asking for
    something more general than what they already gave.
    """
    from app.services.research.engine import apify_social

    handle = (handle or "").strip().lstrip("@")
    platform = (platform or "").strip().lower()
    if not handle:
        return "no handle given."
    if platform not in {"instagram", "tiktok"}:
        return f"cannot read profiles on {platform!r}. Only instagram and tiktok."
    if tc.paid_calls >= tc.max_paid:
        return (
            f"refused: the paid budget for this turn is spent ({tc.paid_calls} of "
            f"{tc.max_paid}). Answer with what you have."
        )

    tc.paid_calls += 1
    token = orchestrator_config(tc).get("APIFY_API_TOKEN")
    if not token:
        return "no Apify token configured, so profiles cannot be read."
    window = ("", "")
    try:
        if platform == "tiktok":
            raw = apify_social.search_tiktok_apify(
                handle, *window, depth="quick", token=token, creators=[handle]
            )
        else:
            raw = apify_social.search_instagram_apify(
                handle, *window, depth="quick", token=token, ig_creators=[handle]
            )
    except Exception as exc:
        logger.warning("tools: profile(%s on %s) failed: %s", handle, platform, exc)
        return f"could not read @{handle} on {platform}: {type(exc).__name__}."

    items = (raw or {}).get("items") or []
    tc.lane_status[platform] = "ok" if items else "no-results"
    if not items:
        error = (raw or {}).get("error")
        return (
            f"@{handle} on {platform} returned no posts"
            + (f" ({error})" if error else ". The account may be private, renamed, or misspelled.")
        )

    followers = next(
        (i.get("author_fans") for i in items if i.get("author_fans")), None
    )
    tags: List[str] = []
    for item in items:
        for tag in item.get("hashtags") or []:
            if tag and tag not in tags:
                tags.append(tag)
    lines = []
    for item in items[:8]:
        text = (item.get("text") or item.get("caption_snippet") or "").strip()
        lines.append(f"- {text[:160]}" if text else "- (no caption)")
    head = f"@{handle} on {platform}: {len(items)} recent posts"
    if followers:
        head += f", {int(followers):,} followers"
    if tags:
        head += f"\nhashtags they use: {', '.join('#' + t for t in tags[:12])}"
    return head + "\nrecent captions:\n" + "\n".join(lines)


def orchestrator_config(tc: ToolContext) -> dict:
    from app.services.grounding import _engine_config

    return _engine_config(tc.settings)


def resolve_account(tc: ToolContext, name: str) -> str:
    """A bare name to a real account, or an honest miss."""
    from app.services.grounding import resolve_seed

    seed = resolve_seed(name, settings=tc.settings)
    if not seed:
        return (
            f"could not work out which account {name!r} is. Matching is strict on "
            "purpose — a confident wrong account is worse than none. Ask the operator "
            "for the handle, or carry on without it."
        )
    return f"{seed.name} is @{seed.handle} on {seed.platform} ({seed.url})"


def extract_creators(tc: ToolContext) -> str:
    """Pull the people out of the pages already searched.

    A tool rather than something the model writes itself: a handle has to come
    from a page that was actually read. Asked to compose them in its reply, a
    model invents plausible ones.
    """
    from app.services.grounding import _drop_news_subjects, extract_creators as extract

    if not tc.findings:
        return "nothing searched yet, so there are no pages to read people out of."
    try:
        found, tags = extract(
            tc.findings, tc.prompt,
            openai_key=tc.settings.openai_api_key,
            model=tc.settings.grounding_model,
            timeout=float(getattr(tc.settings, "search_timeout", 15.0)),
        )
    except Exception as exc:
        logger.warning("tools: creator extraction failed: %s", exc)
        return f"could not read creators out of those pages: {type(exc).__name__}."

    found = _drop_news_subjects(found)
    known = {(c.handle or c.name or "").lower() for c in tc.creators}
    fresh = [c for c in found if (c.handle or c.name or "").lower() not in known]
    tc.creators.extend(fresh)
    tc.hashtags.extend(t for t in tags if t not in tc.hashtags)
    if not fresh:
        return "those pages named nobody new."
    with_handle = sum(1 for c in fresh if c.handle)
    lines = [f"{c.name} {'@' + c.handle if c.handle else '(no handle)'} — {c.why}" for c in fresh[:40]]
    return f"{len(fresh)} people, {with_handle} with a handle:\n" + "\n".join(lines)


def extract_markets(tc: ToolContext) -> str:
    """Pull the countries out, flagged by whether they can be scraped."""
    from app.services.grounding import extract_markets as extract

    if not tc.findings:
        return "nothing searched yet."
    try:
        found = extract(
            tc.findings, tc.prompt, tc.ctx,
            openai_key=tc.settings.openai_api_key,
            model=tc.settings.grounding_model,
            timeout=float(getattr(tc.settings, "search_timeout", 15.0)),
        )
    except Exception as exc:
        return f"could not read markets out of those pages: {type(exc).__name__}."
    tc.markets.extend(found)
    if not found:
        return "those pages did not point at particular countries."
    return "\n".join(
        f"{m.name}{'' if m.supported else ' (NOT scrapeable)'} — {m.why}" for m in found
    )


# ── what the model is shown ──────────────────────────────────────────
#
# The description is the only thing standing between a tool and being used
# wrongly, so each says what it costs and when NOT to reach for it.

def _fn(name: str, description: str, properties: dict, required: List[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


TOOL_SCHEMAS: List[dict] = [
    _fn(
        "search_web",
        "Search the open web. Use the operator's own words rather than keywords — "
        "the search engine reads questions better than a paraphrase of one. This is "
        "the only way to find people who are not already on screen. Costs a search, "
        "so do not repeat a query you have already run.",
        {
            "query": {"type": "string", "description": "What to search for, phrased as the operator asked it"},
            "country": {"type": "string", "description": "ISO-2 code to target, e.g. GH. Omit for no geo-targeting"},
        },
        ["query"],
    ),
    _fn(
        "read_source",
        "Read one search result in full. Search gives you titles and snippets; use "
        "this when a specific page looks like it holds the answer.",
        {"source_id": {"type": "integer", "description": "The id shown in square brackets"}},
        ["source_id"],
    ),
    _fn(
        "search_social",
        "Search Instagram or TikTok for posts, returning real handles and — on "
        "TikTok — follower counts. THIS COSTS MONEY and is refused unless the "
        "operator named that platform. It sweeps a hashtag, so it returns whoever "
        "posted under the tag: good for finding accounts in a niche, useless for "
        "finding one particular person.",
        {
            "platform": {"type": "string", "enum": ["instagram", "tiktok"]},
            "query": {"type": "string", "description": "The topic to search that platform for"},
        },
        ["platform", "query"],
    ),
    _fn(
        "resolve_account",
        "Work out which account a person is, from their name. Use it whenever the "
        "operator names someone and you do not know their handle or which platform "
        "they are on. Returns nothing rather than guessing at a wrong account.",
        {"name": {"type": "string", "description": "The person's name, without an @"}},
        ["name"],
    ),
    _fn(
        "profile",
        "Read a specific account: what it posts, the hashtags it uses, and its "
        "follower count where the platform gives one. THIS COSTS MONEY. Use it "
        "when the operator names an account and the answer depends on what that "
        "account is actually like — finding people similar to it, comparing two of "
        "them, or checking it is who you think. Do this BEFORE searching for people "
        "like someone: a hashtag sweep cannot tell you what they are like.",
        {
            "handle": {"type": "string", "description": "The account handle, with or without the @"},
            "platform": {"type": "string", "enum": ["instagram", "tiktok"]},
        },
        ["handle", "platform"],
    ),
    _fn(
        "extract_creators",
        "Read the people out of the pages you have searched, with their handles "
        "where the pages give them. Call this when the operator wants a list of "
        "people. Never write handles into your own reply — they come from here.",
        {}, [],
    ),
    _fn(
        "extract_markets",
        "Read the countries out of the pages you have searched, flagged by whether "
        "we can scrape there. Call this when the operator asked WHERE rather than who.",
        {}, [],
    ),
    _fn(
        "ask_operator",
        "Stop and ask the operator one question. Use it when the answer depends on "
        "something only they can decide, and searching would guess. Ends the turn.",
        {
            "question": {"type": "string"},
            "options": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Choices to offer, when there are real alternatives",
            },
        },
        ["question"],
    ),
    _fn(
        "answer",
        "Give the operator your answer and end the turn. The lists you extracted "
        "are attached automatically — do not restate them. Say what the evidence "
        "adds up to, and what is missing. If you could not do what was asked, say "
        "so plainly and say what would be needed.",
        {
            "reply": {"type": "string", "description": "Two to five sentences, or a short list when a list IS the answer"},
            "next_step": {"type": "string", "description": "One short question offering the real next step"},
        },
        ["reply"],
    ),
]

TERMINAL = {"ask_operator", "answer"}

_DISPATCH: Dict[str, Callable] = {
    "search_web": search_web,
    "read_source": read_source,
    "search_social": search_social,
    "resolve_account": resolve_account,
    "extract_creators": extract_creators,
    "extract_markets": extract_markets,
    "profile": profile,
}


def run_tool(name: str, arguments: dict, tc: ToolContext) -> str:
    """Run one tool. Never raises: the model reads the failure and carries on."""
    fn = _DISPATCH.get(name)
    if not fn:
        return f"there is no tool called {name!r}."
    try:
        return fn(tc, **(arguments or {}))
    except TypeError as exc:
        return f"wrong arguments for {name}: {exc}"
    except Exception as exc:  # a broken tool must not break the turn
        logger.warning("tools: %s raised %s", name, exc)
        return f"{name} failed: {type(exc).__name__}."
