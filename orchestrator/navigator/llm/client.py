"""Structured calls to Claude, with every failure turned into LLMUnavailable.

    structured(ProfileDraft, system=..., user=...) -> ProfileDraft

- Output is parsed and validated into the given Pydantic model
  (client.beta.messages.parse with output_format=...).
- Server-side refusal fallbacks are on ("default" routing). If the whole chain
  still refuses, the caller gets LLMUnavailable("refusal ...") and the case is
  flagged for a counselor (design.md §7.8).
- The SDK retries rate limits, timeouts and 5xx twice with backoff (E-O4).
  A response that doesn't parse gets one more try (E-O5).
- When the LLM is switched off (NAVIGATOR_LLM=off, or no API key in "auto"),
  every call raises LLMUnavailable("disabled") at once, so each agent uses its
  rule-based fallback. Tests run this way.
"""
from __future__ import annotations

import logging
from typing import Optional, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

from .. import config

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMUnavailable(Exception):
    """The LLM could not produce a usable answer; use the rule-based fallback."""


_client: Optional[anthropic.Anthropic] = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(timeout=config.LLM_TIMEOUT_S, max_retries=2)
    return _client


def structured(output: type[T], *, system: str, user: str, model: Optional[str] = None,
               effort: str = "medium", max_tokens: int = 16000) -> T:
    if not config.llm_enabled():
        raise LLMUnavailable("disabled")
    model = model or config.LLM_MODEL
    last_problem = "no output"
    for _attempt in range(2):
        try:
            resp = _get_client().beta.messages.parse(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_format=output,
                output_config={"effort": effort},
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
        except anthropic.RateLimitError as e:
            raise LLMUnavailable(f"rate limited ({e.status_code})") from e
        except anthropic.APITimeoutError as e:
            raise LLMUnavailable("timed out") from e
        except anthropic.APIConnectionError as e:
            raise LLMUnavailable("could not connect") from e
        except anthropic.APIStatusError as e:
            raise LLMUnavailable(f"API error {e.status_code}") from e
        except ValidationError as e:          # output didn't match the model: try once more
            last_problem = f"invalid structured output ({e.error_count()} errors)"
            continue

        if resp.stop_reason == "refusal":
            category = getattr(resp.stop_details, "category", None) if resp.stop_details else None
            raise LLMUnavailable(f"refusal ({category or 'unspecified'})")
        if resp.stop_reason == "max_tokens":
            last_problem = "output cut off at max_tokens"
            continue
        if resp.parsed_output is not None:
            return resp.parsed_output
        last_problem = "no parsed output"
    log.warning("LLM structured call failed: %s", last_problem)
    raise LLMUnavailable(last_problem)
