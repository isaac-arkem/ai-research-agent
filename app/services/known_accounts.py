# Catalog lookup — named handles already in reference_accounts.
#
# FLOW 3 asks for platform and niche. Niche is job-wide: if any named
# handle already has a topic, that slug covers every handle in the ask.
# Platform can still differ per handle.

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from app.core.supabase import get_supabase_admin

logger = logging.getLogger(__name__)

HANDLE_RE = re.compile(r"@([A-Za-z0-9._]{2,30})")
BARE_AFTER_SCRAPE_RE = re.compile(
    r"(?:scrape|scraping)\s+(?:(?:the|these|this)\s+)?(?:accounts?|profiles?|handles?)?\s*",
    re.IGNORECASE,
)
HANDLE_TOKEN_RE = re.compile(r"[A-Za-z0-9._]{2,30}")
VALID_PLATFORMS = {"tiktok", "instagram"}
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
    """Pull @handles, and bare names after 'scrape', out of operator text."""
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

    scrape_tail = BARE_AFTER_SCRAPE_RE.split(blob, maxsplit=1)
    if len(scrape_tail) > 1:
        for token in re.split(r"[\s,+/&]+", scrape_tail[1]):
            stopped = _clean_handle(token)
            if stopped in {"for", "about", "because"}:
                break
            add(token)

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


def handles_needing_platform(
    prompt: str,
    history: Optional[Iterable] = None,
    known: Optional[Sequence[KnownAccount]] = None,
) -> List[str]:
    """Named handles with no catalog platform and no operator platform."""
    texts = _user_texts(prompt, history)
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
