"""Run one Navigator session in the terminal and print every pause.

    python scripts/demo.py                       # stand-ins; LLM on only if ANTHROPIC_API_KEY is set
    NAVIGATOR_LLM=off python scripts/demo.py     # fully offline
    LENDER_MATCH_URL=http://localhost:8000 python scripts/demo.py   # real Lender Match

Script: Intake asks PUD or condo -> baseline report -> what-if "20% down,
car paid off" -> done -> action plan -> counselor signs off.
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
        print("=" * 78, f"\n[{v['type']} -> {v.get('role')}]")
        if v.get("error"):
            print("!!", v["error"])
        if v["type"] == "intake_questions":
            for q in v["questions"]:
                print(" ?", q["question"])
        elif v["type"] == "counselor_signoff":
            print("Ended:", v["ended_reason"])
            for i, step in enumerate(v["action_plan"], 1):
                print(f" {i}. {step}")
        else:
            print(v.get("report_markdown", v))
    if not out.get("__interrupt__"):
        case = out["case"]
        print("=" * 78, f"\nSession {case['status']} ({case['ended_reason']}), "
              f"{len(case['scenarios'])} scenario(s).")
        print("Events:", " → ".join(e["kind"] for e in case["events"]))


def main():
    graph = build_graph(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "demo"}}
    show(graph.invoke({"session_id": "demo", "profile_input": TOWNHOUSE_MONTGOMERY}, cfg))
    show(graph.invoke(Command(resume={"purchase": {"property_type": "townhouse_pud"}}), cfg))
    show(graph.invoke(Command(resume={
        "action": "what_if", "label": "20% down, car loan paid off",
        "changes": {"purchase.down_payment": 90000, "finances.remove_debt": "car loan"}}), cfg))
    show(graph.invoke(Command(resume={"action": "done"}), cfg))
    show(graph.invoke(Command(resume={"action": "approve", "note": "Reviewed with the borrower."}), cfg))


if __name__ == "__main__":
    main()
