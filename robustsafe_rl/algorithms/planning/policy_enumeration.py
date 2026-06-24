"""Exact deterministic-policy enumeration for tiny finite-model RCMDPs."""

from __future__ import annotations

import itertools
from typing import Dict, Iterator, Optional

import numpy as np

from robustsafe_rl.evaluation import evaluate_policy_over_models


def enumerate_deterministic_policies(
    n_states: int,
    n_actions: int,
) -> Iterator[np.ndarray]:
    """Yield every deterministic policy in lexicographic order."""
    if not isinstance(n_states, (int, np.integer)) or n_states <= 0:
        raise ValueError("n_states must be a positive integer.")
    if not isinstance(n_actions, (int, np.integer)) or n_actions <= 0:
        raise ValueError("n_actions must be a positive integer.")

    for actions in itertools.product(range(int(n_actions)), repeat=int(n_states)):
        yield np.asarray(actions, dtype=int)


def solve_finite_rcmdp_by_enumeration(
    P_models: np.ndarray,
    R_models: np.ndarray,
    C_models: np.ndarray,
    gamma: float,
    cost_threshold: float,
    start_state: int = 0,
) -> Optional[Dict[str, object]]:
    """Return the feasible policy with the largest worst-case reward return."""
    P_models = np.asarray(P_models, dtype=float)
    if P_models.ndim != 4:
        raise ValueError(
            "P_models must have shape [n_models, n_states, n_actions, n_states]."
        )
    if not np.isfinite(cost_threshold):
        raise ValueError("cost_threshold must be finite.")

    n_states, n_actions = P_models.shape[1:3]
    best_policy = None
    best_evaluation = None
    policies_evaluated = 0

    for policy in enumerate_deterministic_policies(n_states, n_actions):
        evaluation = evaluate_policy_over_models(
            P_models,
            R_models,
            C_models,
            policy,
            gamma,
            start_state,
        )
        policies_evaluated += 1
        if evaluation["worst_case_cost"] > cost_threshold:
            continue
        if (
            best_evaluation is None
            or evaluation["worst_case_return"]
            > best_evaluation["worst_case_return"]
        ):
            best_policy = policy.copy()
            best_evaluation = evaluation

    if best_policy is None or best_evaluation is None:
        return None

    return {
        "policy": best_policy,
        **best_evaluation,
        "feasible": True,
        "cost_threshold": float(cost_threshold),
        "policies_evaluated": policies_evaluated,
    }
