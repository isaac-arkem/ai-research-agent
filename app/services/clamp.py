"""Bring out-of-range numbers into range, and say so.

An operator asking for 5,000 posts an account has made an ordinary mistake,
and there are only three things that can happen next:

  the model clamps it     — good, but it is a judgment and it drifted: 900
                            became the cap with a note, while 5,000 came back as
                            a question asking for a smaller number
  the model asks          — a wasted round trip when the answer is obviously
                            "the maximum"
  the validator rejects   — a 422 and a dead turn, and the operator retypes
                            the whole request

Clamping a number to a range is mechanically checkable, so it does not belong
in a prompt at all. This runs before validation, fixes the value, and records
what it did in assumptions so the operator can see it and push back. The
validator stays strict behind it: if this module is doing its job, the
parameter rules should never fire again.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from app.services.validator import VALID_MAX_CREATORS

logger = logging.getLogger(__name__)

MAX_CREATORS_CHOICES = tuple(sorted(VALID_MAX_CREATORS))
POSTS_MIN, POSTS_MAX = 1, 100


def _as_int(value) -> Optional[int]:
    """A number the operator meant, or None. "50" counts; "lots" does not."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _nearest_choice(value: int) -> int:
    """The closest allowed creator count. Ties round up: asked for more, the
    operator is likelier to want more than less."""
    return min(MAX_CREATORS_CHOICES, key=lambda c: (abs(c - value), -c))


def clamp_parameters(parsed: dict) -> Tuple[dict, List[str]]:
    """Fix every out-of-range number in a plan.

    Returns the plan and one note per change, ready to append to assumptions.
    A value that is absent, or not a number at all, is left alone — that is
    the validator's business, not this module's.
    """

    notes: List[str] = []

    def fix_posts(job: dict, where: str) -> None:
        raw = job.get("posts_per_source")
        value = _as_int(raw)
        if value is None or POSTS_MIN <= value <= POSTS_MAX:
            return
        capped = max(POSTS_MIN, min(POSTS_MAX, value))
        job["posts_per_source"] = capped
        notes.append(
            f"Posts per {where} set to {capped}"
            + (f" — {value} is above the {POSTS_MAX} limit." if value > POSTS_MAX
               else f" — {value} is below the minimum of {POSTS_MIN}.")
        )

    for job in parsed.get("recommended_runs") or []:
        if not isinstance(job, dict):
            continue
        fix_posts(job, "source")
        raw = job.get("max_creators")
        value = _as_int(raw)
        if value is not None and value not in MAX_CREATORS_CHOICES:
            nearest = _nearest_choice(value)
            job["max_creators"] = nearest
            notes.append(
                f"Max creators set to {nearest} — {value} is not one of "
                f"{', '.join(str(c) for c in MAX_CREATORS_CHOICES)}."
            )

    for job in parsed.get("reference_accounts") or []:
        if isinstance(job, dict):
            fix_posts(job, "account")

    if notes:
        logger.info("clamped plan parameters: %s", "; ".join(notes))
    return parsed, notes
