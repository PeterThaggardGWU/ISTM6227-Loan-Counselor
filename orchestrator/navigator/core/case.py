"""Read and write the case file held in the graph state.

The state stores the case as a JSON-ready dict (so checkpoints and LangGraph
Studio can show it); nodes convert it to mrn_shared.CaseFile to work on it.
"""
from __future__ import annotations

from typing import Optional

from mrn_shared import CaseFile, Scenario


def load_case(state) -> CaseFile:
    return CaseFile.model_validate(state["case"])


def current(state) -> Scenario:
    return load_case(state).current()


def parent_of(case: CaseFile, scen: Scenario) -> Optional[Scenario]:
    return case.get(scen.parent_id) if scen.parent_id is not None else None


def dump(model) -> dict:
    return model.model_dump(mode="json")
