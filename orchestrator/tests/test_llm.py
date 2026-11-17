"""LLM paths with a fake model: the writer's text is used, the Critic's LLM review runs,
and a refusal falls back to the template and flags the case for a counselor."""
from navigator import config
from navigator.critic.reviewer import Issue, Verdict
from navigator.explainer.writer import NarrativeDraft
from navigator.llm import LLMUnavailable, client


def fake_structured(responses):
    def call(output, **kwargs):
        r = responses.get(output.__name__)
        if isinstance(r, Exception):
            raise r
        return r
    return call


def test_disabled_llm_raises_immediately(monkeypatch):
    monkeypatch.setattr(config, "NAVIGATOR_LLM", "off")
    try:
        client.structured(NarrativeDraft, system="s", user="u")
        raise AssertionError("expected LLMUnavailable")
    except LLMUnavailable as e:
        assert str(e) == "disabled"


def test_llm_narrative_is_used_and_reviewed(session, monkeypatch):
    calls = []
    fake = fake_structured({"NarrativeDraft": NarrativeDraft(paragraphs=["Your odds look solid at 87%."]),
                            "Verdict": Verdict()})
    monkeypatch.setattr("navigator.explainer.writer.structured", lambda o, **k: calls.append("w") or fake(o, **k))
    monkeypatch.setattr("navigator.critic.reviewer.structured", lambda o, **k: calls.append("c") or fake(o, **k))
    s = session()
    rep = s.scenarios[0]["report"]
    assert rep["writer"] == "llm" and "Your odds look solid at 87%." in rep["markdown"]
    assert rep["critic_rounds"][0]["reviewer"] == "rules+llm" and calls == ["w", "c"]


def test_llm_review_issue_sends_the_draft_back(session, monkeypatch):
    verdicts = iter([Verdict(issues=[Issue(kind="jargon", quote="LTV", fix="Explain LTV.")]), Verdict()])
    monkeypatch.setattr("navigator.explainer.writer.structured",
                        lambda o, **k: NarrativeDraft(paragraphs=["Your loan-to-value is explained."]))
    monkeypatch.setattr("navigator.critic.reviewer.structured", lambda o, **k: next(verdicts))
    rep = session().scenarios[0]["report"]
    assert [r["passed"] for r in rep["critic_rounds"]] == [False, True] and rep["revisions"] == 1


def test_refusal_falls_back_to_template_and_escalates(session, monkeypatch):
    monkeypatch.setattr("navigator.explainer.writer.structured",
                        fake_structured({"NarrativeDraft": LLMUnavailable("refusal (cyber)")}))
    s = session()
    assert s.scenarios[0]["report"]["writer"] == "template"
    assert s.pause["type"] == "counselor_review"
    assert any("refusal" in r for r in s.pause["reasons"])
