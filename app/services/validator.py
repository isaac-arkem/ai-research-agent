# Output validation — the "quality gate" for every AI response.
#
# The AI is creative but not always precise. This module checks every
# field in the AI's JSON output against our 9 rules before the operator
# sees it. If anything is wrong, we catch it here with a clear error
# message instead of passing bad data downstream.
#
# Rules:
#   1. Country codes must be from the markets loaded from Supabase
#   2. Pipeline field must match the array it's in
#   3. max_creators must be one of [5, 10, 20, 50, 100, 200]
#   4. posts_per_source must be 1–100; recency_days must be positive or null
#   5. platforms must be "tiktok" and/or "instagram"
#   6. niche must be a lowercase_underscore slug
#   7. hashtags must not be empty
#   8. recommended_runs must have at least one entry (unless off-topic)
#   9. All string fields must be non-empty

import logging
import re
from typing import Dict, List, Set

from app.models.domain import ResearchPlan, ValidationError_, ValidationResult

logger = logging.getLogger(__name__)


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
        if not isinstance(pps, int) or pps < 1 or pps > 100:
            errors.append(ValidationError_(rule=4, field=f"{prefix}.posts_per_source", message=f"posts_per_source must be an integer 1–100, got {pps}"))

        # Rule 5: platforms
        platforms = run.get("platforms", [])
        if not isinstance(platforms, list) or len(platforms) == 0:
            errors.append(ValidationError_(rule=5, field=f"{prefix}.platforms", message="platforms must be a non-empty array"))
        else:
            cleaned_run_plats = []
            for p in platforms:
                plat = str(p).strip().lower()
                if plat not in VALID_PLATFORMS:
                    errors.append(ValidationError_(rule=5, field=f"{prefix}.platforms", message=f'invalid platform "{p}"'))
                elif plat not in cleaned_run_plats:
                    cleaned_run_plats.append(plat)
            run["platforms"] = cleaned_run_plats

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
        if isinstance(recency, str) and recency.strip().lower() in ("any", "null", "none"):
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

        # Rule 9: handles — one or more usernames sharing this job's settings
        handles = ref.get("handles")
        if not handles and ref.get("handle"):
            handles = [ref.get("handle")]
            ref["handles"] = handles
        if not isinstance(handles, list) or len(handles) == 0:
            errors.append(ValidationError_(rule=9, field=f"{prefix}.handles", message="handles must be a non-empty array of usernames"))
        else:
            cleaned = []
            for h in handles:
                if not isinstance(h, str) or not h.strip():
                    errors.append(ValidationError_(rule=9, field=f"{prefix}.handles", message="each handle must be a non-empty string"))
                    break
                cleaned.append(h.strip().lstrip("@"))
            else:
                ref["handles"] = cleaned

        # Rule 5: platforms — array, or legacy singular platform
        platforms = ref.get("platforms")
        if not platforms and ref.get("platform"):
            platforms = [ref.get("platform")]
        if isinstance(platforms, str):
            platforms = [platforms]
        if not isinstance(platforms, list) or len(platforms) == 0:
            errors.append(ValidationError_(rule=5, field=f"{prefix}.platforms", message="platforms must be a non-empty array"))
        else:
            cleaned_plats = []
            for p in platforms:
                plat = str(p).strip().lower()
                if plat not in VALID_PLATFORMS:
                    errors.append(ValidationError_(rule=5, field=f"{prefix}.platforms", message=f'invalid platform "{p}"'))
                elif plat not in cleaned_plats:
                    cleaned_plats.append(plat)
            ref["platforms"] = cleaned_plats
            ref.pop("platform", None)
            pairing = ref.get("handle_platforms")
            if isinstance(pairing, dict):
                cleaned_pair = {}
                for key, value in pairing.items():
                    handle = str(key).strip().lstrip("@")
                    if isinstance(value, list) and value:
                        value = value[0]
                    plat = str(value).strip().lower()
                    if handle and plat in VALID_PLATFORMS:
                        cleaned_pair[handle] = plat
                if len(cleaned_plats) == 1:
                    for h in ref.get("handles") or []:
                        cleaned_pair.setdefault(str(h), cleaned_plats[0])
                if cleaned_pair:
                    ref["handle_platforms"] = cleaned_pair
                else:
                    ref.pop("handle_platforms", None)

        # Rule 6: niche
        niche = ref.get("niche", "")
        if not isinstance(niche, str) or not NICHE_SLUG_RE.match(niche):
            errors.append(ValidationError_(rule=6, field=f"{prefix}.niche", message=f'niche must be a lowercase_underscore slug, got "{niche}"'))

        # Rule 4: posts_per_source (optional, default 10)
        pps = ref.get("posts_per_source")
        if pps is not None:
            if not isinstance(pps, int) or pps < 1 or pps > 100:
                errors.append(ValidationError_(rule=4, field=f"{prefix}.posts_per_source", message=f"posts_per_source must be an integer 1–100, got {pps}"))

        # Rule 4: recency_days (optional)
        recency = ref.get("recency_days")
        if isinstance(recency, str) and recency.strip().lower() in ("any", "null", "none"):
            ref["recency_days"] = None
        elif recency is not None:
            if not isinstance(recency, int) or recency < 1:
                errors.append(ValidationError_(rule=4, field=f"{prefix}.recency_days", message=f'recency_days must be a positive integer, "any", or null, got {recency}'))

        # Rule 9: rationale
        rationale = ref.get("rationale", "")
        if not isinstance(rationale, str) or not rationale.strip():
            errors.append(ValidationError_(rule=9, field=f"{prefix}.rationale", message="rationale must be a non-empty string"))

    if not errors:
        raw["reference_accounts"] = _merge_reference_jobs(refs)

    return _finalise(raw, errors)


def _ref_platforms(ref: dict) -> List[str]:
    platforms = ref.get("platforms")
    if not platforms and ref.get("platform"):
        platforms = [ref.get("platform")]
    if isinstance(platforms, str):
        platforms = [platforms]
    if not isinstance(platforms, list):
        return []
    out = []
    for p in platforms:
        if p in VALID_PLATFORMS and p not in out:
            out.append(p)
    return out


def _merge_reference_jobs(refs: List) -> List:
    """Account scrapes are one job: one row, all handles, all platforms."""
    dicts = [ref for ref in refs if isinstance(ref, dict)]
    others = [ref for ref in refs if not isinstance(ref, dict)]
    if not dicts:
        return refs
    first = {**dicts[0]}
    handles: List = []
    seen_h = set()
    platforms: List[str] = []
    seen_p = set()
    pairing = dict(first.get("handle_platforms") or {}) if isinstance(first.get("handle_platforms"), dict) else {}
    niche = first.get("niche") if isinstance(first.get("niche"), str) else None
    for ref in dicts:
        candidate = ref.get("niche")
        if not niche and isinstance(candidate, str) and candidate.strip():
            niche = candidate.strip()
        hlist = ref.get("handles") if isinstance(ref.get("handles"), list) else []
        plats = _ref_platforms(ref)
        existing_pair = ref.get("handle_platforms") if isinstance(ref.get("handle_platforms"), dict) else {}
        default_plat = plats[0] if len(plats) == 1 else None
        for handle in hlist:
            key = str(handle).lower()
            if key not in seen_h:
                seen_h.add(key)
                handles.append(handle)
            plat = existing_pair.get(handle) or existing_pair.get(key) or default_plat
            if plat:
                pairing[str(handle)] = plat
        for plat in plats:
            if plat not in seen_p:
                seen_p.add(plat)
                platforms.append(plat)
    first["handles"] = handles
    first["platforms"] = platforms
    if niche:
        first["niche"] = niche
    first.pop("platform", None)
    first.pop("handle", None)
    if pairing:
        first["handle_platforms"] = pairing
    return [first, *others]


def _finalise(raw: dict, errors: List[ValidationError_]) -> ValidationResult:
    """Try to parse the raw dict into a ResearchPlan if there are no errors."""
    plan = None
    if not errors:
        try:
            plan = ResearchPlan(**raw)
        except Exception as exc:
            logger.warning("ResearchPlan parse failed after field checks: %s", exc)
            errors.append(
                ValidationError_(
                    rule=0,
                    field="plan",
                    message=f"plan could not be parsed: {exc}",
                )
            )
    return ValidationResult(valid=len(errors) == 0, errors=errors, plan=plan)
