"""Agent nodes. Each reads the current scenario from the case file and returns
only its own result key, so agents that run in parallel never write the same key.

STAND-INS: risk.py, policy.py, explainer.py and critic.py are deterministic
placeholders so the loop runs end to end without LLM calls. lender_match.py calls
the real Lender Match service when LENDER_MATCH_URL is set.
"""
