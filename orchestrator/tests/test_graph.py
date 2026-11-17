"""End-to-end loop with stand-in specialists and the LLM off."""
from navigator.core.routing import agents_to_rerun


def test_baseline_runs_every_agent_then_waits_for_the_borrower(session):
    s = session()
    assert s.pause["type"] == "report" and s.pause["role"] == "borrower"
    scen = s.scenarios[0]
    assert scen["rerun"] == ["lender_match", "risk", "policy"] and scen["kind"] == "baseline"
    assert scen["lender_match"] and scen["risk"] and scen["policy"] and scen["report"]
    assert scen["report"]["status"] == "passed" and scen["report"]["writer"] == "template"
    assert s.case["status"] == "awaiting_borrower"
    assert s.event_kinds()[:2] == ["intake_complete", "ran_agents"]


def test_down_payment_what_if_reruns_all_three_agents(session):
    s = session().resume({"action": "what_if", "changes": {
        "purchase.down_payment": 90000, "finances.remove_debt": "car loan"}})
    base, wi = s.scenarios
    assert wi["rerun"] == ["lender_match", "risk", "policy"] and wi["kind"] == "what_if"
    assert wi["derived"]["ltv_pct"] == 80.0
    assert wi["derived"]["dti_pct"] < base["derived"]["dti_pct"]
    assert not any("mortgage insurance" in c["excerpt"] for c in wi["policy"]["citations"])
    assert "What changed since" in wi["report"]["markdown"]


def test_credit_score_what_if_reruns_only_risk_and_carries_the_rest(session):
    s = session().resume({"action": "what_if", "changes": {"borrower.credit_score": 640}})
    base, wi = s.scenarios
    assert wi["rerun"] == ["risk"]
    assert wi["lender_match"] == base["lender_match"]
    assert wi["policy"] == base["policy"]
    assert wi["risk"]["tier"] == "moderate" and "credit score" in wi["risk"]["top_factors"]


def test_change_that_affects_no_model_reruns_only_the_explainer(session):
    s = session().resume({"action": "what_if", "changes": {"finances.savings": 100000}})
    assert s.scenarios[1]["rerun"] == [] and s.scenarios[1]["report"] is not None


def test_done_goes_to_action_plan_then_counselor_signoff(session):
    s = session().resume({"action": "done"})
    assert s.pause["type"] == "counselor_signoff" and s.pause["role"] == "counselor"
    assert s.case["ended_reason"] == "done"
    assert s.case["action_plan"][-1].startswith("Review this plan with your housing counselor")
    s.resume({"action": "approve", "note": "Discussed credit reports."})
    assert s.pause is None
    assert s.case["status"] == "signed_off"
    assert s.case["counselor_notes"][0]["text"] == "Discussed credit reports."


def test_give_up_still_ends_with_a_plan(session):
    s = session().resume({"action": "give_up"})
    assert s.case["ended_reason"] == "gave_up" and s.case["status"] == "gave_up"
    assert s.pause["type"] == "counselor_signoff" and s.case["action_plan"]


def test_unknown_field_reruns_every_agent():
    assert agents_to_rerun(["purchase.something_new"]) == ["lender_match", "risk", "policy"]
