"""
Software-as-LLM metaphor layer (dual-policy roles, narrative docs).

Pure PPO math lives in ``rl/``; this package holds conceptual mapping and future LLM hooks.
"""

from llm_integration.dual_policy import DualPolicyRoles, describe_dual_policy_roles

__all__ = ["DualPolicyRoles", "describe_dual_policy_roles"]
