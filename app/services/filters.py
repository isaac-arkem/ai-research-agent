"""The operator's filters, read out of the conversation as data.

Filters were scattered: the router put some into fields (country, follower
limits, excluded accounts), the topic text was supposed to carry them into
later turns and often did not, and most were enforced by a model re-reading
the topic sentence. "Not in Ghana" had no field at all, and "no brands" or
"posts in Arabic" had nowhere to go but a hope that the topic kept them.

Here a model reads the whole conversation and writes down every filter in
force for the current request, in a fixed shape. Nothing about any particular
country, size or niche is in the code: the KINDS of filter are fixed, the
values come from what the operator wrote, and anything that does not fit a
kind goes into `requirements` in the operator's own words.

This module only reads filters. Applying them is a separate step.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Sequence

from openai import OpenAI

from app.utils.json_extract import extract_json_object

logger = logging.getLogger(__name__)


@dataclass
class Filters:
    """Every filter in force for one request. Empty fields mean "not set"."""

    countries: List[str] = field(default_factory=list)          # ISO-2, look only here
    exclude_countries: List[str] = field(default_factory=list)  # ISO-2, never these
    min_followers: Optional[int] = None
    max_followers: Optional[int] = None
    platforms: List[str] = field(default_factory=list)          # tiktok | instagram
    exclude_accounts: List[str] = field(default_factory=list)   # handles, no @
    window_days: Optional[int] = None
    # Anything else the operator asked of the results, in their words:
    # "no brands or shops", "women only", "posts in Arabic".
    requirements: List[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not any(asdict(self).values())

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v not in (None, [], "")}


FILTERS_SYSTEM = """You read a research conversation and write down every FILTER the operator has put on the results of their CURRENT request.

A filter limits WHICH results count: where they are from, how big they are, which platform, who to leave out, how recent, or any other condition on them.

NOT FILTERS — never write these down anywhere:
- the subject of the search: the niche, the kind of people to find;
- the account the operator wants others SIMILAR to (the yardstick), or an account they accepted ("yes, use @x"). These are who the search is built around, never accounts to leave out;
- how to compare: "same music style", "same audience", "similar level of fame" — that is the basis of the comparison, not a condition on results.

Return JSON only:
{"countries": ["ISO-2"], "exclude_countries": ["ISO-2"], "min_followers": <int or null>, "max_followers": <int or null>, "platforms": ["tiktok"|"instagram"], "exclude_accounts": ["handle without @"], "window_days": <int or null>, "requirements": ["..."]}

- countries: places the results must be in ("in Saudi Arabia", "UK artists") as ISO-2 codes. exclude_countries: places they must NOT be in ("not in Ghana", "not Ghanaians").
- min_followers / max_followers: the size as a number. "under 100k" -> max 100000. "micro creators", "smaller artists", "not the big names" -> max 100000 unless a number is given. "exclude celebrities" -> max 1000000. "at least 10k" -> min 10000.
- platforms: only platforms they asked for.
- exclude_accounts: ONLY accounts they explicitly said to leave out of the results ("not like @x", "exclude @y", "no @z").
- window_days: "last 30 days" -> 30, "this week" -> 7.
- requirements: every other condition on the results, short and in their words: "no brands or shops", "women only", "posts in Arabic", "no fan pages". A condition already captured by a field above is NOT repeated here: "smaller UK artists" is countries + max_followers, "exclude celebrities" is max_followers — nothing more to add.
- "include brands", "brand accounts too", "pages as well", "add the shops" IS a requirement. Brands and pages are left out unless the operator asks for them, so asking for them is a condition on the results. Write it in their words. A request that never mentions them does not get this line.

THE CONVERSATION DECIDES WHAT IS STILL IN FORCE.
- A message that continues the same request keeps every earlier filter it does not change. "same music style", "instagram", "yes", "look for more" change nothing — carry everything.
- A message that changes a filter replaces only that one: "now do Ghana" replaces the country; "drop the size limit" clears it.
- A message that starts a NEW request ("who is X?", a different topic) carries nothing from before — only what it says itself.

Never invent a filter nobody asked for. A request with no filters returns every field empty."""


def _turn_text(turn) -> str:
    role = getattr(turn, "role", None) or (turn.get("role") if isinstance(turn, dict) else "")
    content = str(getattr(turn, "content", None) or (turn.get("content") if isinstance(turn, dict) else "") or "")
    if role != "assistant":
        return f"Operator: {content.strip()[:600]}"
    try:
        stored = json.loads(content)
    except ValueError:
        return f"Assistant: {content.strip()[:400]}"
    if not isinstance(stored, dict):
        return f"Assistant: {str(stored)[:400]}"
    said = " ".join(str(stored.get(k) or "").strip() for k in ("understood_so_far", "clarifying_question"))
    return f"Assistant: {said.strip()[:500]}"


def _codes(value) -> List[str]:
    out: List[str] = []
    for raw in value if isinstance(value, list) else []:
        code = str(raw or "").strip().upper()
        if len(code) == 2 and code.isalpha() and code not in out:
            out.append(code)
    return out


def _count(value) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = int(float(str(value).replace(",", "").strip()))
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _words(value, *, lower: bool = False, strip_at: bool = False) -> List[str]:
    out: List[str] = []
    for raw in value if isinstance(value, list) else []:
        text = str(raw or "").strip()
        if strip_at:
            text = text.lstrip("@")
        if lower:
            text = text.lower()
        if text and text.lower() not in ("null", "none", "n/a") and text not in out:
            out.append(text)
    return out


def parse_filters(parsed: dict) -> Filters:
    """The model's JSON as Filters, keeping only well-formed values."""
    parsed = parsed if isinstance(parsed, dict) else {}
    return Filters(
        countries=_codes(parsed.get("countries")),
        exclude_countries=_codes(parsed.get("exclude_countries")),
        min_followers=_count(parsed.get("min_followers")),
        max_followers=_count(parsed.get("max_followers")),
        platforms=[p for p in _words(parsed.get("platforms"), lower=True) if p in ("tiktok", "instagram")],
        exclude_accounts=_words(parsed.get("exclude_accounts"), lower=True, strip_at=True),
        window_days=_count(parsed.get("window_days")),
        requirements=_words(parsed.get("requirements")),
    )


def extract_filters(
    prompt: str,
    history: Optional[Sequence] = None,
    *,
    settings,
    keep: int = 8,
    previous: Optional[dict] = None,
) -> Optional[Filters]:
    """Every filter in force for the request this message belongs to.

    None when the model could not be asked — callers carry on without filters,
    never with guessed ones.
    """
    turns = [_turn_text(t) for t in list(history or [])[-keep:]]
    context = (
        ("Conversation so far:\n" + "\n".join(turns) + "\n\n" if turns else "No earlier turns.\n\n")
        # What was applied to the last answer. The conversation may be longer
        # than the turns shown above; this is how a filter set ten messages
        # ago is still known.
        + (f"Filters applied to the last answer: {json.dumps(previous, ensure_ascii=False)}\n\n"
           if previous else "")
        + f"Latest message: {prompt.strip()}"
    )
    try:
        client = OpenAI(api_key=settings.openai_api_key,
                        timeout=float(getattr(settings, "search_timeout", 15.0)))
        response = client.chat.completions.create(
            model=getattr(settings, "filters_model", "gpt-4o-mini"),
            messages=[{"role": "system", "content": FILTERS_SYSTEM},
                      {"role": "user", "content": context}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        parsed = extract_json_object(response.choices[0].message.content or "") or {}
    except Exception as exc:
        logger.warning("filter extraction failed: %s", exc)
        return None
    return parse_filters(parsed)
