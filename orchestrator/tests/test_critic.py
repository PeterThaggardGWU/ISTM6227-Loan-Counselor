"""Critic rules catch each kind of problem; a clean template report passes."""
from mrn_shared import Citation, Report
from navigator.critic.rules import check_text, review
from navigator.explainer.template import DISCLAIMER

GOOD = (f"**Can you get approved?** GREEN: odds are 87%.\n**Can you sustain the payments?** GREEN\n"
        f"Loan amount $382,500.\n{DISCLAIMER}")
FACTS = {"approval_odds": "87%", "loan_amount": "$382,500"}


def report(md=GOOD, citations=()):
    return Report(approval_signal="green", sustainability_signal="green", markdown=md,
                  facts=FACTS, citations=list(citations))


def test_clean_report_passes():
    assert review(report(), []) == []


def test_decision_language_is_flagged():
    assert check_text("Good news: you will be approved!")
    assert check_text("You're pre-approved")
    assert check_text("you qualify for this loan")
    assert not check_text("Your estimated approval odds are 87%.")


def test_steering_and_protected_traits_are_flagged():
    assert check_text("We recommend Lender 2.")
    assert check_text("As a single mom you should be careful.")


def test_numbers_not_in_the_facts_are_flagged():
    fixes = review(report(GOOD.replace("$382,500", "$400,000")), [])
    assert any("400000" in f for f in fixes)


def test_missing_disclaimer_and_wrong_signal_line_are_flagged():
    fixes = review(report(GOOD.replace(DISCLAIMER, "").replace("approved?** GREEN", "approved?** RED")), [])
    assert any("disclaimer" in f for f in fixes) and any("must show GREEN" in f for f in fixes)


def test_invented_citation_is_flagged():
    c = Citation(source="Fannie Mae Selling Guide", excerpt="made up")
    assert any("citation" in f for f in review(report(citations=[c]), []))


def test_every_baseline_template_report_passes(session):
    assert session().scenarios[0]["report"]["critic_rounds"][0]["passed"]
