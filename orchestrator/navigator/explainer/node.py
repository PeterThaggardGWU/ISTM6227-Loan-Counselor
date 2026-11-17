"""The Explainer node: store this round's results, compute signals and facts,
write the narrative (LLM, or template on any LLM failure) and compose the draft."""
from __future__ import annotations

from mrn_shared import LenderMatchResult, PolicyResult, Report, RiskResult

from ..core.case import dump, load_case, parent_of
from ..llm import LLMUnavailable
from .facts import build_facts
from .signals import approval_signal, sustainability_signal
from .template import compose, template_narrative
from .writer import write_narrative

_RESULTS = (("lender_match", LenderMatchResult), ("risk", RiskResult), ("policy", PolicyResult))


def attach_results(state, case):
    """Store this round's specialist results on the scenario; carried-over results stay."""
    scen = case.current()
    for key, model in _RESULTS:
        if key in scen.rerun and state.get(key):
            setattr(scen, key, model.model_validate(state[key]))
    return scen


def explainer(state):
    case = load_case(state)
    scen = attach_results(state, case)
    parent = parent_of(case, scen)
    approval, sustain = approval_signal(scen), sustainability_signal(scen)
    facts = build_facts(scen, parent, approval, sustain)
    citations = list(scen.policy.citations) if scen.policy else []

    previous = state.get("draft") or {}
    fixes = (state.get("critic") or {}).get("fixes", []) if previous else []
    flags = list(state.get("llm_flags") or [])
    writer = "llm"
    try:
        narrative = write_narrative(facts, {"approval": approval.rule, "sustainability": sustain.rule},
                                    citations, fixes)
    except LLMUnavailable as e:
        writer, narrative = "template", template_narrative(facts)
        if str(e) != "disabled":
            flags.append(f"Explainer fell back to the template ({e})")

    report = Report(approval_signal=approval.signal, sustainability_signal=sustain.signal,
                    markdown=compose(scen, approval, sustain, facts, narrative, citations),
                    citations=citations, facts=facts, writer=writer,
                    critic_rounds=previous.get("critic_rounds", []))
    return {"case": dump(case), "draft": dump(report), "llm_flags": flags}
