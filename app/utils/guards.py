# Input sanitisation — the "security checkpoint" for operator prompts.
#
# Before the operator's question reaches the AI, it passes through here.
# We clean up invisible characters, collapse extra whitespace, and enforce
# a length ceiling. This prevents:
#   - Hidden control characters that could confuse the LLM
#   - BiDi override characters (used in text-direction attacks)
#   - A megabyte paste that wastes tokens and money
#
# The ceiling is high on purpose: operators often paste a brief — a few
# paragraphs of situation — before the actual ask. 2000 chars was clipping
# that. 32k is several pages, which is the brief; it is not unlimited.

import re
import unicodedata

MAX_PROMPT_LENGTH = 32_000

# Matches invisible control characters (ASCII 0x00–0x08, 0x0B, 0x0C, 0x0E–0x1F, 0x7F)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Matches Unicode BiDi override characters — these can make text appear
# different from what's actually there (a classic spoofing trick)
_BIDI_RE = re.compile(r"[‎‏‪-‮⁦-⁩]")

# Matches any run of whitespace (spaces, tabs, newlines) for collapsing
_WHITESPACE_RE = re.compile(r"\s+")


def sanitize_prompt(raw: str) -> str:
    """Clean up an operator's raw input before sending it to the LLM.

    Steps:
      1. Normalise Unicode (e.g. ligatures → standard chars)
      2. Strip invisible control characters
      3. Strip BiDi override characters
      4. Collapse all whitespace to single spaces
      5. Trim leading/trailing whitespace
      6. Cap at MAX_PROMPT_LENGTH (a brief, not a document dump)
    """
    text = unicodedata.normalize("NFKC", raw)
    text = _CONTROL_RE.sub("", text)
    text = _BIDI_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text[:MAX_PROMPT_LENGTH]
