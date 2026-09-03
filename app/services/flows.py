# Flow classification — determines the operator flow from the plan's shape.
#
# Three operator flows (prompt stories in app/services/prompt.py):
#   1. discovery      — "Find new creators" → creator_intelligence
#   2. deep_research  — "Go deeper / compare markets" → creator_intelligence
#                       (one run per country, larger max_creators)
#   3. reference      — "Scrape these accounts" → reference_profiles

from typing import Optional

from app.models.domain import FlowName, ResearchPlan


def classify_flow(plan: Optional[ResearchPlan]) -> FlowName:
    if plan is None:
        return "off_topic"

    runs = plan.recommended_runs
    refs = plan.reference_accounts
    if not runs and not refs:
        return "off_topic"
    if refs and not runs:
        return "reference"
    if refs and runs:
        return "mixed"

    countries = {c for run in runs for c in run.countries}
    max_creators = max((run.max_creators for run in runs), default=0)
    if len(runs) > 1 or len(countries) > 1 or max_creators >= 100:
        return "deep_research"
    return "discovery"
