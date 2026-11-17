"""The guard around Peter's agents: failures and timeouts become 'unavailable'."""
import time

from navigator.core.specialists import guarded


def boom(state):
    raise RuntimeError("service down")


def slow(state):
    time.sleep(1)
    return {"risk": {}}


def test_exception_becomes_unavailable():
    out = guarded("lender_match", boom)({})
    assert out["lender_match"]["source"] == "unavailable"


def test_timeout_becomes_unavailable():
    out = guarded("risk", slow, timeout_s=0.05)({})
    assert out["risk"]["source"] == "unavailable" and out["risk"]["assessed"] is False


def test_session_survives_a_failed_specialist(session):                  # E-O2
    s = session(specialists={"risk": boom})
    assert s.pause["type"] == "report"
    assert s.pause["sustainability_signal"] == "not_assessed"


def test_all_services_down_escalates_without_fake_numbers(session):      # E-O3
    s = session(specialists={"risk": boom, "lender_match": boom})
    assert s.pause["type"] == "counselor_review"
    assert s.scenarios[0]["report"]["approval_signal"] == "not_assessed"
