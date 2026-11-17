"""Apply what-if changes to a profile, returning a new, validated profile."""
from __future__ import annotations

from typing import Any

from .schema import BorrowerProfile

# Special operations besides plain "dotted.path": value
REMOVE_DEBT = "finances.remove_debt"   # value: debt name (case-insensitive), e.g. "car loan"


def apply_changes(profile: BorrowerProfile, changes: dict[str, Any]) -> BorrowerProfile:
    data = profile.model_dump(mode="json")
    for path, value in changes.items():
        if path == REMOVE_DEBT:
            name = str(value).strip().lower()
            debts = data["finances"]["monthly_debts"]
            kept = [d for d in debts if d["name"].strip().lower() != name]
            if len(kept) == len(debts):
                raise KeyError(f"No debt named {value!r}")
            data["finances"]["monthly_debts"] = kept
            continue
        node = data
        *parents, leaf = path.split(".")
        for key in parents:
            node = node[key]
        if leaf not in node:
            raise KeyError(f"Unknown profile field: {path}")
        node[leaf] = value
    return BorrowerProfile.model_validate(data)   # re-validates every field and the cross-field rules
