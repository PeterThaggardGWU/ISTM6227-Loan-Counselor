from __future__ import annotations

from mrn_shared import CaseFile, Scenario


def load_case(state) -> CaseFile:
    return CaseFile.model_validate(state["case"])


def current(state) -> Scenario:
    return load_case(state).current()


def dump(model) -> dict:
    return model.model_dump(mode="json")
