"""Shared fixtures: no network, no LLM, stand-in specialists, fixed rate."""
import copy
import sys
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from navigator import config  # noqa: E402
from navigator.graph import build_graph  # noqa: E402
from sample_profiles import TOWNHOUSE_MONTGOMERY_PUD  # noqa: E402


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(config, "LENDER_MATCH_URL", "")
    monkeypatch.setattr(config, "ASSUMED_RATE_PCT", 6.5)
    monkeypatch.setattr(config, "NAVIGATOR_LLM", "off")


class Session:
    """Drive one graph thread: start it, resume it, read the latest pause."""

    def __init__(self, profile=None, specialists=None, **extra):
        self.graph = build_graph(checkpointer=InMemorySaver(), specialists=specialists)
        self.cfg = {"configurable": {"thread_id": "t"}}
        start = {"session_id": "t", "profile_input": copy.deepcopy(profile or TOWNHOUSE_MONTGOMERY_PUD), **extra}
        self.out = self.graph.invoke(start, self.cfg)

    def resume(self, value):
        self.out = self.graph.invoke(Command(resume=value), self.cfg)
        return self

    @property
    def pause(self):
        intr = self.out.get("__interrupt__")
        return intr[0].value if intr else None

    @property
    def case(self):
        return self.out["case"]

    @property
    def scenarios(self):
        return self.out["case"]["scenarios"]

    def event_kinds(self):
        return [e["kind"] for e in self.case["events"]]


@pytest.fixture
def session():
    return Session
