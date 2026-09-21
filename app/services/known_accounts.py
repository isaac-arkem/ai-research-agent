# Catalog lookup — named handles already in reference_accounts.
#
# FLOW 3 asks for platform and niche. Niche is job-wide: if any named
# handle already has a topic, that slug covers every handle in the ask.
# Platform can still differ per handle.

from __future__ import annotations

import logging
import json
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from app.core.supabase import get_supabase_admin
from app.services.validator import VALID_PLATFORMS

logger = logging.getLogger(__name__)

HANDLE_RE = re.compile(r"@([A-Za-z0-9._]{2,30})")
HANDLE_TOKEN_RE = re.compile(r"[A-Za-z0-9._]{2,30}")
SKIP_TOKENS = {
    "on",
    "and",
    "the",
    "for",
    "from",
    "with",
    "please",
    "this",
    "that",
    "these",
    "those",
    "tiktok",
    "instagram",
    "insta",
    "ig",
    "tt",
    "account",
    "accounts",
    "profile",
    "profiles",
    "handle",
    "handles",
    "scrape",
    "scraping",
}

# Words that never appear inside a list of account names, and so mark where
# the list ended and the sentence resumed. SKIP_TOKENS are passed over ("and",
# "on Instagram"); these stop the walk outright.
NICHE_SLUG_RE = re.compile(r"^[a-z0-9_]+$")
PLATFORM_WORD_RE = re.compile(
    r"\b(tiktok|instagram|insta|reels)\b|\b(?:ig|tt)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class KnownAccount:
    handle: str
    platform: str
    niche: Optional[str]


def _clean_handle(raw: str) -> str:
    return raw.strip().lstrip("@").rstrip(".,;:").lower()


def extract_handles(*texts: str) -> List[str]:
    """Pull the @handles out of operator text. Literally the @ ones.

    This used to also guess at BARE names after the word "scrape" — walking
    the words that followed and deciding, from two hand-written lists, which
    were names and which were English. That is a judgement about what the
    operator meant, and it was made by a word list.

    It got it wrong in the way judgements-by-word-list always do: "scrape
    sarkodie's specific profiles so find his handles" became the accounts
    ['specific', 'so', 'find', 'his']. Four ordinary words promoted to
    handles, which flipped the turn into a named-account job, which bypasses
    research entirely — and then every reply for the rest of the thread was
    read as answering a question about those four accounts. Sarkodie's
    handles were asked for eight times and never found.

    Who the operator named is now the router's "subjects", decided by reading
    the sentence. This does the part that needs no judgement: an @ is an @.
    """
    found: List[str] = []
    seen = set()

    def add(token: str) -> None:
        handle = _clean_handle(token)
        if len(handle) < 2 or handle in SKIP_TOKENS or handle in seen:
            return
        if not HANDLE_TOKEN_RE.fullmatch(handle):
            return
        seen.add(handle)
        found.append(handle)

    blob = " ".join(t for t in texts if t)
    for match in HANDLE_RE.finditer(blob):
        add(match.group(1))

    return found


def _slug_niche(raw: Optional[str]) -> Optional[str]:
    if not raw or not str(raw).strip():
        return None
    slug = str(raw).strip().lower().replace(" ", "_")
    slug = re.sub(r"[^a-z0-9_]", "", slug)
    if not slug or not NICHE_SLUG_RE.match(slug):
        return None
    return slug


def _handle_variants(handles: Sequence[str]) -> List[str]:
    variants = []
    seen = set()
    for handle in handles:
        cleaned = _clean_handle(handle)
        if not cleaned:
            continue
        for item in (cleaned, cleaned.lower(), f"@{cleaned}", f"@{cleaned.lower()}"):
            if item not in seen:
                seen.add(item)
                variants.append(item)
    return variants


def lookup_known_accounts(
    handles: Sequence[str],
    *,
    client=None,
) -> List[KnownAccount]:
    """Return catalog rows for these handles. Empty on any failure."""
    variants = _handle_variants(handles)
    if not variants:
        return []

    db = client if client is not None else get_supabase_admin()
    if db is None:
        return []

    try:
        result = (
            db.table("reference_accounts")
            .select("handle,platform,topic")
            .in_("handle", variants)
            .execute()
        )
        rows = result.data or []
    except Exception as exc:
        logger.warning("reference_accounts lookup failed: %s", exc)
        return []

    wanted = {h.lower() for h in variants}
    found: List[KnownAccount] = []
    seen = set()
    for row in rows:
        handle = _clean_handle(str(row.get("handle") or ""))
        platform = str(row.get("platform") or "").strip().lower()
        if handle not in wanted or platform not in VALID_PLATFORMS:
            continue
        key = (handle, platform)
        if key in seen:
            continue
        seen.add(key)
        found.append(
            KnownAccount(
                handle=handle,
                platform=platform,
                niche=_slug_niche(row.get("topic")),
            )
        )
    return found


def shared_job_niche(accounts: Sequence[KnownAccount]) -> Optional[str]:
    """First catalog niche wins for the whole named-account job."""
    for account in accounts:
        if account.niche:
            return account.niche
    return None


def _user_texts(prompt: str, history: Optional[Iterable] = None) -> List[str]:
    texts = [prompt]
    for turn in history or []:
        role = getattr(turn, "role", None) or (
            turn.get("role") if isinstance(turn, dict) else None
        )
        content = getattr(turn, "content", None) or (
            turn.get("content") if isinstance(turn, dict) else None
        )
        if role == "user" and content:
            texts.append(str(content))
    return texts


def operator_named_platform(*texts: str) -> bool:
    """True when the operator already said TikTok or Instagram."""
    return any(PLATFORM_WORD_RE.search(text or "") for text in texts)


def _role_of(turn) -> Optional[str]:
    return getattr(turn, "role", None) or (
        turn.get("role") if isinstance(turn, dict) else None
    )


def _content_of(turn) -> str:
    content = getattr(turn, "content", None) or (
        turn.get("content") if isinstance(turn, dict) else None
    )
    return str(content or "")


# Fields whose question has nothing to do with which accounts to scrape.
_COMPARISON_RE = re.compile(
    r"\b(?:similar|similarly|similar\s+to|lookalikes?|look-alikes?|"
    r"comparable|compares?|comparison|competitors?|alternatives?|"
    # An account offered AS a reference, example, benchmark or yardstick is
    # the thing "similar" is measured against — never the thing to scrape.
    r"references?|examples?|benchmarks?|yardsticks?|"
    r"resembl\w+|in\s+the\s+style\s+of|same\s+(?:style|vibe|kind|sort)\s+as)\b"
    r"|(?<!would )(?<!should )(?<!'d )\blike\b(?!\s+to\b)",
    re.IGNORECASE,
)


def accounts_are_references(text: str) -> bool:
    """Does this message name accounts as a COMPARISON, not as the job?

    When it does, the handles are inputs to a search — the seeds — and the
    answer is other people entirely. Treating them as the job is what turned
    "Find creators similar to @sarkodie and @shattawale, then rank them by
    similarity and engagement rate" into a plan to scrape those two accounts.
    """
    return bool(text) and bool(_COMPARISON_RE.search(text))


def handles_needing_platform(
    prompt: str,
    history: Optional[Iterable] = None,
    known: Optional[Sequence[KnownAccount]] = None,
) -> List[str]:
    """Named handles with no catalog platform and no operator platform."""
    if accounts_are_references(prompt):
        # The platform of a SEED does not gate anything: we are not scraping
        # it. Asking for it here stalls the search behind an irrelevant field.
        return []
    texts = _user_texts(prompt, history)
    # The handles are often not in THIS message. "same audience" — the whole
    # of a reply that picks how to compare — names nobody, and the handles it
    # inherits come from the question two turns back. Read on its own it looks
    # like a bare answer inside a named-account job, so the platform question
    # fired and grounding was skipped: picking an option ended the search
    # instead of refining it.
    #
    # So when this message names nobody, the turn that DID name them decides.
    # A message carrying its own handles is still judged on its own merits,
    # which keeps "scrape @a and @b" later in the same thread a real job.
    if not extract_handles(prompt) and any(accounts_are_references(t) for t in texts):
        return []
    named = extract_handles(*texts)
    known_h = {account.handle.lower() for account in (known or [])}
    unknown = [handle for handle in named if handle not in known_h]
    if not unknown or operator_named_platform(*texts):
        return []
    return unknown


def platform_clarifying_question(handles: Sequence[str]) -> str:
    labels = [f"@{handle}" for handle in handles if handle]
    if not labels:
        return "Which platform should I scrape — TikTok or Instagram?"
    if len(labels) == 1:
        return f"Which platform is {labels[0]} on — TikTok or Instagram?"
    listed = ", ".join(labels[:-1]) + f" and {labels[-1]}"
    return (
        f"Which platform are {listed} on — TikTok or Instagram? "
        "They can be different."
    )


def known_accounts_from_text(
    prompt: str,
    history: Optional[Iterable] = None,
    *,
    client=None,
) -> List[KnownAccount]:
    texts = _user_texts(prompt, history)
    return lookup_known_accounts(extract_handles(*texts), client=client)
