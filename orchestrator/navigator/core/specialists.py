"""The guard between the Orchestrator and the specialists (Peter's agents).

Contract for a specialist: fn(state) -> {"<name>": <result dict>}, where the
result is a LenderMatchResult, RiskResult or PolicyResult from mrn_shared.

guarded() wraps one so that it can never stop the session:
- an exception or a timeout returns an "unavailable" result instead;
- the report then says "not assessed" rather than showing made-up numbers.
LangGraph waits for every parallel branch, so this per-call timeout is what
gives the join its timeout (design.md §7.7).
"""
from __future__ import annotations

import concurrent.futures as cf
import logging
from typing import Callable

from mrn_shared import LenderMatchResult, PolicyResult, RiskResult

from .. import config
from .case import dump

log = logging.getLogger(__name__)

_UNAVAILABLE = {
    "lender_match": lambda why: LenderMatchResult(source="unavailable", note=why),
    "risk": lambda why: RiskResult(source="unavailable", assessed=False, note=why),
    "policy": lambda why: PolicyResult(source="unavailable"),
}

# Shared pool: a timed-out call keeps running in its thread, but the graph moves on.
_POOL = cf.ThreadPoolExecutor(max_workers=8, thread_name_prefix="specialist")


def unavailable(name: str, why: str) -> dict:
    return {name: dump(_UNAVAILABLE[name](why))}


def guarded(name: str, fn: Callable[[dict], dict], timeout_s: float | None = None):
    def node(state):
        limit = timeout_s if timeout_s is not None else config.SPECIALIST_TIMEOUT_S
        future = _POOL.submit(fn, state)
        try:
            out = future.result(timeout=limit)
        except cf.TimeoutError:
            log.warning("%s timed out after %ss", name, limit)
            return unavailable(name, f"{name} did not answer within {limit:g} s")
        except Exception as e:  # noqa: BLE001 - any failure becomes "unavailable"
            log.exception("%s failed", name)
            return unavailable(name, f"{name} failed ({type(e).__name__})")
        if not isinstance(out, dict) or name not in out:
            return unavailable(name, f"{name} returned no result")
        return out

    node.__name__ = name
    return node


def specialist_status(scen) -> dict[str, str]:
    """source of each specialist result on a scenario: service, stub, unavailable or missing."""
    return {name: (getattr(scen, name).source if getattr(scen, name) else "missing")
            for name in _UNAVAILABLE}
