# Drop unsupported countries from a plan instead of failing it.
#
# WHY THIS EXISTS
# ---------------
# Operators ask by region — "the Gulf", "MENA", "Southeast Asia". The model
# expands the region from its own geography, which is the right behaviour, but
# a real region contains countries our table does not: MENA includes Jordan,
# the Gulf includes Qatar and Bahrain. Whether any given one is in the table
# depends on what the scrape actor supports that week.
#
# Before this, one unlisted code failed the WHOLE plan — the operator asked for
# MENA and got "Validation failed: JO is not a supported market" instead of the
# ten MENA countries we can actually scrape. That is a bad trade: the plan was
# fine apart from one country nobody explicitly asked for.
#
# So we filter rather than reject. This runs BEFORE validation, so the plan the
# validator sees is already clean.
#
# WHAT IT DOES NOT DO
# -------------------
# It does not silently swallow the operator's actual request. Two guards:
#
#   * Every dropped country is named in `assumptions`. The operator sees
#     exactly what was left out, so a missing market is never a mystery.
#   * If filtering empties the plan entirely — every country they asked for is
#     unsupported — it does NOT return a cheerful empty plan. It leaves the
#     plan empty and puts the explanation in `risks`, which is the existing
#     unsupported-market contract.
#
# The prompt also tells the model to self-filter. That instruction stays: it
# saves a round trip when it works. This is the backstop for when it does not,
# and it is deterministic where the instruction is not.

from __future__ import annotations

import logging
from typing import List, Set, Tuple

logger = logging.getLogger(__name__)


def filter_unsupported_countries(
    parsed: dict, valid_codes: Set[str]
) -> Tuple[dict, List[str]]:
    """Strip unsupported country codes from recommended_runs.

    Returns the plan and the sorted list of codes removed. A run left with no
    countries is dropped entirely — a scrape with no market cannot run.

    Only touches recommended_runs: reference_accounts are named handles and
    carry no country, by design (FLOW 3 never asks for one).
    """
    runs = parsed.get("recommended_runs")
    if not isinstance(runs, list) or not runs:
        return parsed, []

    dropped: Set[str] = set()
    kept_runs = []

    for run in runs:
        if not isinstance(run, dict):
            kept_runs.append(run)
            continue

        countries = run.get("countries")
        if not isinstance(countries, list):
            kept_runs.append(run)
            continue

        supported = []
        for code in countries:
            if not isinstance(code, str):
                continue
            if code in valid_codes:
                supported.append(code)
            else:
                dropped.add(code)

        if not supported:
            # Nothing left to scrape. Dropping the run beats emitting one with
            # an empty countries array, which fails validation anyway.
            logger.info(
                "dropping run %r — no supported countries left", run.get("title")
            )
            continue

        kept_runs.append({**run, "countries": supported})

    if not dropped:
        return parsed, []

    plan = {**parsed, "recommended_runs": kept_runs}
    removed = sorted(dropped)

    assumptions = plan.get("assumptions")
    plan["assumptions"] = list(assumptions) if isinstance(assumptions, list) else []
    plan["assumptions"].append(
        "Not scraping " + ", ".join(removed) + ": "
        + ("that country is" if len(removed) == 1 else "those countries are")
        + " not in the supported list. The rest of the request was planned as asked."
    )

    # Everything was dropped — say so in risks rather than returning a plan
    # that looks successful but scrapes nothing.
    if not kept_runs and not plan.get("reference_accounts"):
        risks = plan.get("risks")
        plan["risks"] = list(risks) if isinstance(risks, list) else []
        plan["risks"].append(
            "None of the countries in this request are supported ("
            + ", ".join(removed)
            + "), so there is nothing to scrape. Pick a country from the "
            "supported list and I can plan it."
        )

    logger.info("filtered unsupported countries: %s", removed)
    return plan, removed
