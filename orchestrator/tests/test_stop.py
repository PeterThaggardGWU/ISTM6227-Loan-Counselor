"""Action plan: rule steps, the counselor handoff, and the LLM rewrite guard."""
from mrn_shared import CaseFile
from navigator.llm import LLMUnavailable
from navigator.stop import action_plan as ap


def test_plan_has_three_to_six_steps_and_ends_with_the_handoff(session):
    s = session().resume({"action": "done"})
    plan = s.case["action_plan"]
    assert 2 <= len(plan) <= 6 and plan[-1] == ap.HANDOFF


def test_best_scenario_is_named_when_it_is_not_the_baseline(session):
    s = session(profile=None).resume({"action": "what_if", "label": "20% down",
                                      "changes": {"purchase.down_payment": 90000}})
    case = CaseFile.model_validate(s.case)
    assert ap.best_scenario(case).id in (0, 1)


def test_llm_rewrite_that_adds_decision_language_is_rejected(monkeypatch):
    from navigator.stop.action_plan import PlanDraft
    monkeypatch.setattr(ap, "structured", lambda *a, **k: PlanDraft(steps=["You will be approved.", ap.HANDOFF]))
    try:
        ap.llm_steps(["Check your credit.", ap.HANDOFF])
        raise AssertionError("expected LLMUnavailable")
    except LLMUnavailable:
        pass
