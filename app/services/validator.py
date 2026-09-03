# Output validation — the "quality gate" for every AI response.
#
# The AI is creative but not always precise. This module checks every
# field in the AI's JSON output against our 9 rules before the operator
# sees it. If anything is wrong, we catch it here with a clear error
# message instead of passing bad data downstream.
#
# Rules:
#   1. Country codes must be from our 19 supported markets
#   2. Pipeline field must match the array it's in
#   3. max_creators must be one of [5, 10, 20, 50, 100, 200]
#   4. posts_per_source must be 1–200; recency_days must be positive or null
#   5. platforms must be "tiktok" and/or "instagram"
#   6. niche must be a lowercase_underscore slug
#   7. hashtags must not be empty
#   8. recommended_runs must have at least one entry (unless off-topic)
#   9. All string fields must be non-empty

import re
from typing import List, Set

from app.models.domain import ResearchPlan, ValidationError_, ValidationResult


VALID_MAX_CREATORS = {5, 10, 20, 50, 100, 200}
VALID_PLATFORMS = {"tiktok", "instagram"}
NICHE_SLUG_RE = re.compile(r"^[a-z0-9_]+$")


def validate_research_plan(
    raw: dict,
    valid_country_codes: Set[str],
) -> ValidationResult:
    """Check every field in the AI's output against our rules.
    Returns a ValidationResult with either the validated plan or a list of errors."""

    errors: List[ValidationError_] = []

    if not isinstance(raw, dict):
        return ValidationResult(
            valid=False,
            errors=[ValidationError_(rule=0, field="root", message="Not a JSON object")],
        )

    # ── Clarifying question — pass through without plan validation ──
    if "clarifying_question" in raw:
        cq = raw["clarifying_question"]
        if isinstance(cq, str) and cq.strip():
            return ValidationResult(valid=True, errors=[], plan=None)
        return ValidationResult(
            valid=False,
            errors=[ValidationError_(rule=9, field="clarifying_question", message="clarifying_question must be a non-empty string")],
        )

    # ── Top-level fields ─────────────────────────────────────────

    if not isinstance(raw.get("summary"), str) or not raw["summary"].strip():
        errors.append(ValidationError_(rule=9, field="summary", message="summary must be a non-empty string"))

    for field in ("assumptions", "patterns_to_watch", "content_angles", "risks"):
        if not isinstance(raw.get(field), list):
            errors.append(ValidationError_(rule=9, field=field, message=f"{field} must be an array"))

    # ── Off-topic detection ──────────────────────────────────────
    # An empty plan is valid ONLY if the AI included guidance in risks.
    # Otherwise it means the AI just didn't try.

    runs = raw.get("recommended_runs", [])
    if not isinstance(runs, list):
        runs = []

    refs = raw.get("reference_accounts", [])
    if not isinstance(refs, list):
        refs = []

    is_off_topic = len(runs) == 0 and len(refs) == 0

    if is_off_topic:
        risks = raw.get("risks", [])
        if not isinstance(risks, list) or len(risks) == 0:
            errors.append(ValidationError_(rule=8, field="recommended_runs", message="Empty plan must include guidance in risks"))
        return _finalise(raw, errors)

    # Reference-only plan (e.g. "scrape @khloekardashian") is valid
    if len(runs) == 0 and len(refs) == 0:
        errors.append(ValidationError_(rule=8, field="recommended_runs", message="recommended_runs must have at least one entry"))

    # ── Validate each recommended run ────────────────────────────

    for i, run in enumerate(runs):
        prefix = f"recommended_runs[{i}]"

        if not isinstance(run, dict):
            errors.append(ValidationError_(rule=0, field=prefix, message="run must be an object"))
            continue

        # Rule 2: pipeline
        if run.get("pipeline") != "creator_intelligence":
            errors.append(ValidationError_(rule=2, field=f"{prefix}.pipeline", message=f'pipeline must be "creator_intelligence", got "{run.get("pipeline")}"'))

        # Rule 1: country codes
        countries = run.get("countries", [])
        if not isinstance(countries, list) or len(countries) == 0:
            errors.append(ValidationError_(rule=1, field=f"{prefix}.countries", message="countries must be a non-empty array"))
        else:
            for code in countries:
                if code not in valid_country_codes:
                    errors.append(ValidationError_(rule=1, field=f"{prefix}.countries", message=f'"{code}" is not a supported market'))

        # Rule 3: max_creators
        if run.get("max_creators") not in VALID_MAX_CREATORS:
            errors.append(ValidationError_(rule=3, field=f"{prefix}.max_creators", message=f"max_creators must be one of [5,10,20,50,100,200], got {run.get('max_creators')}"))

        # Rule 4: posts_per_source
        pps = run.get("posts_per_source")
        if not isinstance(pps, int) or pps < 1 or pps > 200:
            errors.append(ValidationError_(rule=4, field=f"{prefix}.posts_per_source", message=f"posts_per_source must be an integer 1–200, got {pps}"))

        # Rule 5: platforms
        platforms = run.get("platforms", [])
        if not isinstance(platforms, list) or len(platforms) == 0:
            errors.append(ValidationError_(rule=5, field=f"{prefix}.platforms", message="platforms must be a non-empty array"))
        else:
            for p in platforms:
                if p not in VALID_PLATFORMS:
                    errors.append(ValidationError_(rule=5, field=f"{prefix}.platforms", message=f'invalid platform "{p}"'))

        # Rule 6: niche slug
        niche = run.get("niche", "")
        if not isinstance(niche, str) or not NICHE_SLUG_RE.match(niche):
            errors.append(ValidationError_(rule=6, field=f"{prefix}.niche", message=f'niche must be a lowercase_underscore slug, got "{niche}"'))

        # Rule 7: hashtags
        hashtags = run.get("hashtags", [])
        if not isinstance(hashtags, list) or len(hashtags) < 3:
            errors.append(ValidationError_(rule=7, field=f"{prefix}.hashtags", message="hashtags must have at least 3 entries"))

        # Rule 9: non-empty strings
        for str_field in ("title", "rationale"):
            val = run.get(str_field, "")
            if not isinstance(val, str) or not val.strip():
                errors.append(ValidationError_(rule=9, field=f"{prefix}.{str_field}", message=f"{str_field} must be a non-empty string"))

        # Rule 4: recency_days (if present)
        recency = run.get("recency_days")
        if recency == "any":
            run["recency_days"] = None
        elif recency is not None:
            if not isinstance(recency, int) or recency < 1:
                errors.append(ValidationError_(rule=4, field=f"{prefix}.recency_days", message=f"recency_days must be a positive integer, \"any\", or null, got {recency}"))

    # ── Validate each reference account ──────────────────────────

    for i, ref in enumerate(refs):
        prefix = f"reference_accounts[{i}]"

        if not isinstance(ref, dict):
            errors.append(ValidationError_(rule=0, field=prefix, message="ref must be an object"))
            continue

        # Rule 2: pipeline
        if ref.get("pipeline") != "reference_profiles":
            errors.append(ValidationError_(rule=2, field=f"{prefix}.pipeline", message=f'pipeline must be "reference_profiles", got "{ref.get("pipeline")}"'))

        # Rule 9: handle
        handle = ref.get("handle", "")
        if not isinstance(handle, str) or not handle.strip():
            errors.append(ValidationError_(rule=9, field=f"{prefix}.handle", message="handle must be a non-empty string"))

        # Rule 5: platform
        if ref.get("platform") not in VALID_PLATFORMS:
            errors.append(ValidationError_(rule=5, field=f"{prefix}.platform", message=f'invalid platform "{ref.get("platform")}"'))

        # Rule 6: niche
        niche = ref.get("niche", "")
        if not isinstance(niche, str) or not NICHE_SLUG_RE.match(niche):
            errors.append(ValidationError_(rule=6, field=f"{prefix}.niche", message=f'niche must be a lowercase_underscore slug, got "{niche}"'))

        # Rule 4: posts_per_source (optional, default 10)
        pps = ref.get("posts_per_source")
        if pps is not None:
            if not isinstance(pps, int) or pps < 1 or pps > 200:
                errors.append(ValidationError_(rule=4, field=f"{prefix}.posts_per_source", message=f"posts_per_source must be an integer 1–200, got {pps}"))

        # Rule 4: recency_days (optional)
        recency = ref.get("recency_days")
        if recency == "any":
            ref["recency_days"] = None
        elif recency is not None:
            if not isinstance(recency, int) or recency < 1:
                errors.append(ValidationError_(rule=4, field=f"{prefix}.recency_days", message=f'recency_days must be a positive integer, "any", or null, got {recency}'))

        # Rule 9: rationale
        rationale = ref.get("rationale", "")
        if not isinstance(rationale, str) or not rationale.strip():
            errors.append(ValidationError_(rule=9, field=f"{prefix}.rationale", message="rationale must be a non-empty string"))

    return _finalise(raw, errors)


def _finalise(raw: dict, errors: List[ValidationError_]) -> ValidationResult:
    """Try to parse the raw dict into a ResearchPlan if there are no errors."""
    plan = None
    if not errors:
        try:
            plan = ResearchPlan(**raw)
        except Exception:
            pass
    return ValidationResult(valid=len(errors) == 0, errors=errors, plan=plan)
