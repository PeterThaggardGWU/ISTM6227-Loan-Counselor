"""Intake: questions, consistency checks, free text, the turn cap."""
import copy

from navigator import config
from navigator.intake import extractor
from navigator.intake.questions import MAX_QUESTIONS, next_questions
from sample_profiles import TOWNHOUSE_MONTGOMERY, TOWNHOUSE_MONTGOMERY_PUD


def test_missing_field_is_asked_then_the_loop_continues(session):
    p = copy.deepcopy(TOWNHOUSE_MONTGOMERY_PUD)
    del p["purchase"]["target_state"]
    s = session(p)
    assert s.pause["type"] == "intake_questions"
    assert any(q["field"] == "purchase.target_state" for q in s.pause["questions"])
    s.resume({"purchase": {"target_state": "MD"}})
    assert s.pause["type"] == "report"


def test_plain_townhouse_asks_pud_or_condo(session):                     # E-I5
    s = session(TOWNHOUSE_MONTGOMERY)
    assert [q["field"] for q in s.pause["questions"]] == ["purchase.property_type"]
    s.resume({"purchase": {"property_type": "townhouse_condo"}})
    assert s.scenarios[0]["profile"]["purchase"]["property_type"] == "townhouse_condo"


def test_county_outside_the_service_area_is_explained(session):         # E-I7
    p = copy.deepcopy(TOWNHOUSE_MONTGOMERY_PUD)
    p["purchase"].update(target_state="TX", target_county_fips="48029", target_county_name="Bexar County")
    s = session(p)
    assert "DC metro area only" in s.pause["questions"][0]["question"]


def test_high_dti_needs_a_confirmation_once(session):                   # E-I3
    p = copy.deepcopy(TOWNHOUSE_MONTGOMERY_PUD)
    p["finances"]["annual_income"] = 60000
    s = session(p)
    assert [q["field"] for q in s.pause["questions"]] == ["confirm.high_dti"]
    s.resume({"confirm": {"high_dti": True}})
    assert s.pause["type"] == "report"


def test_at_most_three_questions_per_turn():
    problems = [{"field": f"f{i}", "problem": "x", "kind": "missing"} for i in range(6)]
    assert len(next_questions(problems)) == MAX_QUESTIONS


def test_free_text_without_llm_asks_for_fields(session):
    p = copy.deepcopy(TOWNHOUSE_MONTGOMERY_PUD)
    del p["purchase"]["target_state"]
    s = session(p).resume("I'm buying in Maryland")
    assert "field by field" in s.pause["error"]


def test_free_text_answer_is_read_by_the_llm(session, monkeypatch):
    p = copy.deepcopy(TOWNHOUSE_MONTGOMERY_PUD)
    del p["purchase"]["target_state"]
    monkeypatch.setattr("navigator.intake.node.extract_profile",
                        lambda text, raw, asked: {"purchase": {"target_state": "MD"}})
    s = session(p).resume("I'm buying in Maryland")
    assert s.pause["type"] == "report"


def test_turn_cap_hands_the_questions_to_a_counselor(session, monkeypatch):   # E-I15
    monkeypatch.setattr(config, "MAX_INTAKE_TURNS", 1)
    p = copy.deepcopy(TOWNHOUSE_MONTGOMERY_PUD)
    del p["purchase"]["target_state"]
    s = session(p)
    assert s.pause["role"] == "borrower"
    s.resume({"borrower": {"first_time_buyer": True}})      # dodges the question
    assert s.pause["role"] == "counselor"


def test_profile_draft_has_no_protected_trait_fields():                 # E-I9
    banned = {"race", "ethnicity", "sex", "gender", "age", "marital_status", "religion",
              "national_origin", "disability", "familial_status"}
    assert not banned & set(extractor.ProfileDraft.model_fields)
