from app.services.flows import classify_flow
from tests.plans import deep_research_plan, discovery_plan, reference_plan


def test_flow_1_discovery():
    plan = discovery_plan()
    assert classify_flow(plan) == "discovery"


def test_flow_2_deep_research():
    plan = deep_research_plan()
    assert classify_flow(plan) == "deep_research"


def test_flow_3_reference():
    plan = reference_plan()
    assert classify_flow(plan) == "reference"


def test_mixed_plan():
    plan = discovery_plan()
    plan.reference_accounts = reference_plan().reference_accounts
    assert classify_flow(plan) == "mixed"
