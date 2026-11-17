"""Specialist agents: Lender Match, Risk and Policy (Peter's; the gray box).

Contract: fn(state) -> {"<name>": <result dict>} with a LenderMatchResult,
RiskResult or PolicyResult from mrn_shared. The graph wraps each one in
core.specialists.guarded(), so a failure or timeout becomes an "unavailable"
result instead of stopping the session.

The non-gray agents live in their own packages: intake/, explainer/, critic/,
human/ (borrower and counselor) and stop/.
"""
