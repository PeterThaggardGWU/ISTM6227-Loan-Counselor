"""The Orchestrator graph.

    START → intake → [lender_match ∥ risk ∥ policy]  (only the agents a change affects)
          → explainer ⇄ critic                         (revise until the critic passes, max MAX_REVISIONS)
          → publish_report → await_borrower            (pauses for the borrower: interrupt)
          → what_if → apply_what_if → [affected agents] → … (loops)
          → done / give_up / MAX_ROUNDS → finish → END

`graph` is what `langgraph dev` and Studio load (they supply the checkpointer).
Scripts and tests call build_graph(checkpointer=InMemorySaver()).
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from . import config
from .agents.borrower import apply_what_if, await_borrower, finish, publish_report
from .agents.critic import critic
from .agents.explainer import explainer
from .agents.intake import intake
from .agents.lender_match import lender_match
from .agents.policy import policy
from .agents.risk import risk
from .routing import AGENTS
from .state import NavState


def route_agents(state) -> list[str]:
    """Fan out to every agent this round needs; they run in parallel."""
    return list(state.get("to_run") or []) or ["explainer"]


def route_after_apply(state):
    return "await_borrower" if state.get("reply_error") else route_agents(state)


def route_after_critic(state) -> str:
    passed = (state.get("critic") or {}).get("passed", False)
    return "publish_report" if passed or state.get("revisions", 0) >= config.MAX_REVISIONS else "explainer"


def route_after_publish(state) -> str:
    return "finish" if len(state["case"]["scenarios"]) >= config.MAX_ROUNDS else "await_borrower"


def route_after_reply(state) -> str:
    action = (state.get("reply") or {}).get("action")
    if action == "what_if":
        return "apply_what_if"
    if action == "invalid":
        return "await_borrower"
    return "finish"


def build_graph(checkpointer=None):
    g = StateGraph(NavState)
    for name, fn in [("intake", intake), ("lender_match", lender_match), ("risk", risk),
                     ("policy", policy), ("explainer", explainer), ("critic", critic),
                     ("publish_report", publish_report), ("await_borrower", await_borrower),
                     ("apply_what_if", apply_what_if), ("finish", finish)]:
        g.add_node(name, fn)

    targets = [*AGENTS, "explainer"]
    g.add_edge(START, "intake")
    g.add_conditional_edges("intake", route_agents, targets)
    for agent in AGENTS:
        g.add_edge(agent, "explainer")                  # join: explainer runs once all finish
    g.add_edge("explainer", "critic")
    g.add_conditional_edges("critic", route_after_critic, ["explainer", "publish_report"])
    g.add_conditional_edges("publish_report", route_after_publish, ["await_borrower", "finish"])
    g.add_conditional_edges("await_borrower", route_after_reply, ["apply_what_if", "await_borrower", "finish"])
    g.add_conditional_edges("apply_what_if", route_after_apply, [*targets, "await_borrower"])
    g.add_edge("finish", END)
    return g.compile(checkpointer=checkpointer)


graph = build_graph()
