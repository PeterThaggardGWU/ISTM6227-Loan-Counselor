"""The Orchestrator graph: wiring only. Each node comes from one package.

    START → intake ⟲ (questions)                         intake/
          → [lender_match ∥ risk ∥ policy]                agents/ (Peter), wrapped by core/specialists
          → explainer ⇄ critic   (≤ MAX_REVISIONS)        explainer/, critic/
          → publish_report → counselor_review?            human/ (escalation rules: core/escalation)
          → await_borrower → apply_what_if → [affected specialists] → …   (core/routing)
          → done / give up / max rounds → action_plan → counselor_signoff → END   (stop/, core/stopping)

Every routing decision below is a plain rule; the LLM never chooses an edge (D7).
`graph` is what `langgraph dev` and Studio load (they supply the checkpointer).
Scripts and tests call build_graph(checkpointer=InMemorySaver()).
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from . import config
from .agents.lender_match import lender_match
from .agents.policy import policy
from .agents.risk import risk
from .core.case import load_case
from .core.routing import AGENTS
from .core.specialists import guarded
from .core.stopping import max_rounds_reached
from .critic.node import critic
from .explainer.node import explainer
from .human.borrower import apply_what_if, await_borrower, publish_report
from .human.counselor import counselor_review, counselor_signoff
from .intake.node import intake
from .state import NavState
from .stop.node import action_plan

SPECIALISTS = {"lender_match": lender_match, "risk": risk, "policy": policy}


def route_agents(state) -> list[str]:
    """Fan out to every specialist this round needs; they run in parallel."""
    return list(state.get("to_run") or []) or ["explainer"]


def route_after_intake(state):
    return route_agents(state) if state.get("intake_done") else "intake"


def route_after_apply(state):
    return "await_borrower" if state.get("reply_error") else route_agents(state)


def route_after_critic(state) -> str:
    passed = (state.get("critic") or {}).get("passed", False)
    return "publish_report" if passed or state.get("revisions", 0) >= config.MAX_REVISIONS else "explainer"


def route_after_publish(state) -> str:
    if state.get("escalation"):
        return "counselor_review"
    return "action_plan" if max_rounds_reached(load_case(state)) else "await_borrower"


def route_after_review(state) -> str:
    if not state.get("counselor_done"):
        return "counselor_review"
    return "action_plan" if max_rounds_reached(load_case(state)) else "await_borrower"


def route_after_reply(state) -> str:
    action = (state.get("reply") or {}).get("action")
    if action == "what_if":
        return "apply_what_if"
    if action == "invalid":
        return "await_borrower"
    return "action_plan"


def route_after_signoff(state) -> str:
    return END if state.get("counselor_done") else "counselor_signoff"


def build_graph(checkpointer=None, specialists: dict | None = None):
    """specialists: override Peter's agents (tests use this to simulate failures)."""
    fns = {**SPECIALISTS, **(specialists or {})}
    g = StateGraph(NavState)
    g.add_node("intake", intake)
    for name in AGENTS:
        g.add_node(name, guarded(name, fns[name]))
    for name, fn in [("explainer", explainer), ("critic", critic), ("publish_report", publish_report),
                     ("counselor_review", counselor_review), ("await_borrower", await_borrower),
                     ("apply_what_if", apply_what_if), ("action_plan", action_plan),
                     ("counselor_signoff", counselor_signoff)]:
        g.add_node(name, fn)

    targets = [*AGENTS, "explainer"]
    g.add_edge(START, "intake")
    g.add_conditional_edges("intake", route_after_intake, [*targets, "intake"])
    for agent in AGENTS:
        g.add_edge(agent, "explainer")                  # join: explainer runs once all finish
    g.add_edge("explainer", "critic")
    g.add_conditional_edges("critic", route_after_critic, ["explainer", "publish_report"])
    g.add_conditional_edges("publish_report", route_after_publish,
                            ["counselor_review", "await_borrower", "action_plan"])
    g.add_conditional_edges("counselor_review", route_after_review,
                            ["counselor_review", "await_borrower", "action_plan"])
    g.add_conditional_edges("await_borrower", route_after_reply, ["apply_what_if", "await_borrower", "action_plan"])
    g.add_conditional_edges("apply_what_if", route_after_apply, [*targets, "await_borrower"])
    g.add_edge("action_plan", "counselor_signoff")
    g.add_conditional_edges("counselor_signoff", route_after_signoff, ["counselor_signoff", END])
    return g.compile(checkpointer=checkpointer)


graph = build_graph()
