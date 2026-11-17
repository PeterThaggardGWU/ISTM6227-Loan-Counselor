"""Borrower and counselor nodes: what-if edge cases, escalation, review."""
from mrn_shared import RiskResult
from navigator import config


def test_noop_what_if_creates_no_scenario(session):                      # B9 / E-W1
    s = session().resume({"action": "what_if", "changes": {"purchase.down_payment": 67500}})
    assert "same as the current scenario" in s.pause["error"] and len(s.scenarios) == 1


def test_invalid_reply_explains_itself(session):                         # B10 / E-W6
    s = session().resume("maybe?")
    assert "didn't understand" in s.pause["error"]


def test_non_editable_field_is_rejected(session):                        # E-W5
    s = session().resume({"action": "what_if", "changes": {"borrower.current_state": "VA"}})
    assert "can't be changed" in s.pause["error"]


def test_down_payment_above_savings_is_rejected(session):
    s = session().resume({"action": "what_if", "changes": {"purchase.down_payment": 95000}})
    assert "savings" in s.pause["error"] and len(s.scenarios) == 1


def test_branch_from_an_earlier_scenario(session):                       # E-W9
    s = session().resume({"action": "what_if", "changes": {"purchase.down_payment": 90000}})
    s.resume({"action": "what_if", "from_scenario": 0, "changes": {"borrower.credit_score": 760}})
    assert s.scenarios[2]["parent_id"] == 0
    assert s.scenarios[2]["profile"]["purchase"]["down_payment"] == 67500


def test_free_text_what_if_uses_the_llm_parse(session, monkeypatch):
    from navigator.intake.extractor import ChangeDraft, WhatIfDraft
    monkeypatch.setattr("navigator.human.borrower.parse_what_if", lambda text, prof: WhatIfDraft(
        label="20% down", changes=[ChangeDraft(path="purchase.down_payment", number=90000)]))
    s = session().resume({"action": "what_if", "text": "what if I put 20% down?"})
    assert s.scenarios[1]["label"] == "20% down" and s.scenarios[1]["derived"]["ltv_pct"] == 80.0


def test_vague_what_if_asks_instead_of_guessing(session, monkeypatch):
    from navigator.intake.extractor import WhatIfDraft
    monkeypatch.setattr("navigator.human.borrower.parse_what_if",
                        lambda text, prof: WhatIfDraft(label="raise", unparsed="a raise next year"))
    s = session().resume({"action": "what_if", "text": "what if I get a raise next year?"})
    assert "a raise next year" in s.pause["error"] and len(s.scenarios) == 1


def elevated_risk(state):
    return {"risk": RiskResult(source="service", assessed=True, tier="elevated").model_dump(mode="json")}


def test_red_sustainability_goes_to_a_counselor_first(session):
    s = session(specialists={"risk": elevated_risk})
    assert s.pause["type"] == "counselor_review"
    assert any("Sustainability is red" in r for r in s.pause["reasons"])
    s.resume({"action": "annotate", "note": "Talk about reserves."})
    assert s.pause["type"] == "report" and s.pause["counselor_reviewed"]
    assert s.case["counselor_notes"][0]["text"] == "Talk about reserves."


def test_counselor_edit_keeps_the_original(session):
    s = session(specialists={"risk": elevated_risk}).resume({"action": "edit", "markdown": "Edited.", "note": "x"})
    rep = s.scenarios[0]["report"]
    assert rep["markdown"] == "Edited." and rep["original_markdown"].startswith("## Your readiness")


def test_report_failing_the_critic_never_reaches_the_borrower(session, monkeypatch):   # B2 / D15
    monkeypatch.setattr("navigator.critic.node.review", lambda report, cites: ["always wrong"])
    s = session()
    assert s.pause["type"] == "counselor_review"
    assert s.scenarios[0]["report"]["status"] == "needs_review"
    assert len(s.scenarios[0]["report"]["critic_rounds"]) == config.MAX_REVISIONS


def test_max_rounds_ends_with_a_reason_and_plan(session, monkeypatch):   # B11 / E-W10
    monkeypatch.setattr(config, "MAX_ROUNDS", 2)
    s = session().resume({"action": "what_if", "changes": {"borrower.credit_score": 760}})
    assert s.pause["type"] == "counselor_signoff"
    assert s.case["ended_reason"] == "max_rounds"


def test_rounds_without_progress_offer_a_wrap_up(session):               # E-W12
    s = session()
    for score in (720, 725):
        s.resume({"action": "what_if", "changes": {"borrower.credit_score": score}})
    assert s.pause["suggest_wrap_up"]
    s.resume({"action": "done"})
    assert s.case["ended_reason"] == "no_progress"
