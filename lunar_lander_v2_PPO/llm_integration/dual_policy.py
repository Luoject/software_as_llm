"""
Maps PPO's behavior / training policies to the software-as-LLM dual-agent metaphor.

- ``policy_old``: rollout (inference) policy — collects trajectories without gradient updates.
- ``policy``: learner policy — optimized with clipped surrogate loss, then synced to ``policy_old``.

This file documents intent only; training logic remains in ``rl.ppo.PPO``.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DualPolicyRoles:
    """Names used in docs when relating PPO to lead/sub-agent narratives."""

    rollout_policy_attr: str = "policy_old"
    learner_policy_attr: str = "policy"
    rollout_role: str = "subagent"
    learner_role: str = "lead_agent"


def describe_dual_policy_roles() -> str:
    roles = DualPolicyRoles()
    return (
        f"PPO uses two networks: {roles.rollout_policy_attr} ({roles.rollout_role}) "
        f"collects experience; {roles.learner_policy_attr} ({roles.learner_role}) "
        "is updated and periodically copied to the rollout network."
    )
