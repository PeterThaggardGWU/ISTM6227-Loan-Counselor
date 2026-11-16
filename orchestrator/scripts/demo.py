"""Run one Navigator session in the terminal: baseline, one what-if, then stop.

    python scripts/demo.py                  # stand-in agents
    LENDER_MATCH_URL=http://localhost:8000 python scripts/demo.py   # real Lender Match

No LLM calls are made by the current stand-in agents.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from navigator.graph import build_graph
from sample_profiles import TOWNHOUSE_MONTGOMERY


def show(out):
    for intr in out.get("__interrupt__", []):
        v = intr.value
        print("=" * 78)
        if v.get("error"):
            print("!!", v["error"])
        print(v.get("report_markdown", v))
    if not out.get("__interrupt__"):
        print("=" * 78, f"\nSession finished: {out['case']['status']} "
              f"after {len(out['case']['scenarios'])} scenario(s).")


def main():
    graph = build_graph(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "demo"}}
    show(graph.invoke({"session_id": "demo", "profile_input": TOWNHOUSE_MONTGOMERY}, cfg))
    show(graph.invoke(Command(resume={
        "action": "what_if", "label": "20% down, car loan paid off",
        "changes": {"purchase.down_payment": 90000, "finances.remove_debt": "car loan"}}), cfg))
    final = graph.invoke(Command(resume={"action": "done"}), cfg)
    show(final)
    for s in final["case"]["scenarios"]:
        print(f"  scenario {s['id']} '{s['label']}': reran {s['rerun'] or 'nothing'}")


if __name__ == "__main__":
    main()
