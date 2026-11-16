import copy
import sys
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from mrn_shared import Report  # noqa: E402
from navigator import config  # noqa: E402
from navigator.agents.critic import review  # noqa: E402
from navigator.graph import build_graph  # noqa: E402
from navigator.routing import agents_to_rerun  # noqa: E402
from sample_profiles import TOWNHOUSE_MONTGOMERY  # noqa: E402


@pytest.fixture(autouse=True)
def stand_ins(monkeypatch):
    monkeypatch.setattr(config, "LENDER_MATCH_URL", "")
    monkeypatch.setattr(config, "ASSUMED_RATE_PCT", 6.5)


def start(profile=TOWNHOUSE_MONTGOMERY):
    graph = build_graph(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t"}}
    out = graph.invoke({"session_id": "t", "profile_input": copy.deepcopy(profile)}, cfg)
    return graph, cfg, out


def test_baseline_runs_every_agent_then_waits_for_the_borrower():
    _, _, out = start()
    assert out["__interrupt__"][0].value["type"] == "report"
    scen = out["case"]["scenarios"][0]
    assert scen["rerun"] == ["lender_match", "risk", "policy"]
    assert scen["lender_match"] and scen["risk"] and scen["policy"] and scen["report"]
    assert out["case"]["status"] == "awaiting_borrower"


def test_down_payment_what_if_reruns_all_three_agents():
    graph, cfg, out = start()
    out = graph.invoke(Command(resume={"action": "what_if", "changes": {
        "purchase.down_payment": 90000, "finances.remove_debt": "car loan"}}), cfg)
    base, wi = out["case"]["scenarios"]
    assert wi["rerun"] == ["lender_match", "risk", "policy"]     # LTV changes the MI guidance
    assert wi["derived"]["ltv_pct"] == 80.0
    assert wi["derived"]["dti_pct"] < base["derived"]["dti_pct"]
    assert not any("mortgage insurance" in c["excerpt"] for c in wi["policy"]["citations"])
    assert "What changed since" in wi["report"]["markdown"]


def test_credit_score_what_if_reruns_only_risk_and_carries_the_rest():
    graph, cfg, out = start()
    out = graph.invoke(Command(resume={"action": "what_if", "changes": {"borrower.credit_score": 640}}), cfg)
    base, wi = out["case"]["scenarios"]
    assert wi["rerun"] == ["risk"]
    assert wi["lender_match"] == base["lender_match"]           # carried over, not rerun
    assert wi["policy"] == base["policy"]
    assert wi["risk"]["tier"] == "moderate" and "credit score" in wi["risk"]["top_factors"]


def test_change_that_affects_no_model_reruns_only_the_explainer():
    graph, cfg, _ = start()
    out = graph.invoke(Command(resume={"action": "what_if", "changes": {"finances.savings": 100000}}), cfg)
    assert out["case"]["scenarios"][1]["rerun"] == []
    assert out["case"]["scenarios"][1]["report"] is not None


def test_bad_what_if_is_reported_and_the_borrower_is_asked_again():
    graph, cfg, _ = start()
    out = graph.invoke(Command(resume={"action": "what_if", "changes": {"purchase.down_payment": 500000}}), cfg)
    v = out["__interrupt__"][0].value
    assert "Couldn't apply" in v["error"]
    assert len(out["case"]["scenarios"]) == 1


def test_done_finishes_the_session():
    graph, cfg, _ = start()
    out = graph.invoke(Command(resume={"action": "done"}), cfg)
    assert "__interrupt__" not in out
    assert out["case"]["status"] == "done"


def test_missing_profile_fields_trigger_intake_questions():
    profile = copy.deepcopy(TOWNHOUSE_MONTGOMERY)
    del profile["purchase"]["target_state"]
    graph, cfg, out = start(profile)
    v = out["__interrupt__"][0].value
    assert v["type"] == "intake_questions"
    assert any(p["field"] == "purchase.target_state" for p in v["problems"])
    out = graph.invoke(Command(resume={"purchase": {"target_state": "MD"}}), cfg)
    assert out["__interrupt__"][0].value["type"] == "report"


def test_critic_flags_decision_language_and_missing_disclaimer():
    bad = Report(approval_signal="green", sustainability_signal="green",
                 markdown="Good news: you will be approved!")
    fixes = review(bad, policy_citation_count=1)
    assert len(fixes) == 3


def test_unknown_field_reruns_every_agent():
    assert agents_to_rerun(["purchase.something_new"]) == ["lender_match", "risk", "policy"]
