"""Exact discounted evaluation for deterministic tabular policies."""

from __future__ import annotations

from typing import Dict, Tuple, Union

import numpy as np


def _validate_single_model(
    P: np.ndarray,
    R: np.ndarray,
    C: np.ndarray,
    policy: np.ndarray,
    gamma: float,
    start_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    P = np.asarray(P, dtype=float)
    R = np.asarray(R, dtype=float)
    C = np.asarray(C, dtype=float)
    policy = np.asarray(policy)

    if P.ndim != 3 or P.shape[0] != P.shape[2]:
        raise ValueError("P must have shape [n_states, n_actions, n_states].")
    n_states, n_actions, _ = P.shape
    if R.shape != (n_states, n_actions) or C.shape != (n_states, n_actions):
        raise ValueError("R and C must have shape [n_states, n_actions].")
    if policy.shape != (n_states,) or not np.issubdtype(policy.dtype, np.integer):
        raise ValueError("policy must be an integer array with shape [n_states].")
    if np.any(policy < 0) or np.any(policy >= n_actions):
        raise ValueError("policy contains an invalid action.")
    if not 0.0 <= gamma < 1.0:
        raise ValueError("gamma must satisfy 0 <= gamma < 1.")
    if not isinstance(start_state, (int, np.integer)) or not 0 <= start_state < n_states:
        raise ValueError("start_state is out of range.")
    if not np.allclose(P.sum(axis=-1), 1.0):
        raise ValueError("Each transition-probability row must sum to 1.")
    if np.any(P < 0.0):
        raise ValueError("Transition probabilities cannot be negative.")

    return P, R, C, policy.astype(int, copy=False)


def evaluate_deterministic_policy(
    P: np.ndarray,
    R: np.ndarray,
    C: np.ndarray,
    policy: np.ndarray,
    gamma: float,
    start_state: int = 0,
) -> Tuple[float, float]:
    """Return exact discounted reward and cost from ``start_state``."""
    P, R, C, policy = _validate_single_model(
        P, R, C, policy, gamma, start_state
    )
    states = np.arange(P.shape[0])
    policy_transitions = P[states, policy]
    policy_rewards = R[states, policy]
    policy_costs = C[states, policy]
    system = np.eye(P.shape[0]) - gamma * policy_transitions

    reward_values = np.linalg.solve(system, policy_rewards)
    cost_values = np.linalg.solve(system, policy_costs)
    return float(reward_values[start_state]), float(cost_values[start_state])


def evaluate_policy_over_models(
    P_models: np.ndarray,
    R_models: np.ndarray,
    C_models: np.ndarray,
    policy: np.ndarray,
    gamma: float,
    start_state: int = 0,
) -> Dict[str, Union[float, np.ndarray]]:
    """Evaluate one policy under every model and aggregate robust metrics."""
    P_models = np.asarray(P_models, dtype=float)
    R_models = np.asarray(R_models, dtype=float)
    C_models = np.asarray(C_models, dtype=float)

    if P_models.ndim != 4 or P_models.shape[1] != P_models.shape[3]:
        raise ValueError(
            "P_models must have shape [n_models, n_states, n_actions, n_states]."
        )
    expected_shape = P_models.shape[:3]
    if R_models.shape != expected_shape or C_models.shape != expected_shape:
        raise ValueError(
            "R_models and C_models must have shape [n_models, n_states, n_actions]."
        )
    if P_models.shape[0] == 0:
        raise ValueError("At least one model is required.")

    values = [
        evaluate_deterministic_policy(
            P_models[index],
            R_models[index],
            C_models[index],
            policy,
            gamma,
            start_state,
        )
        for index in range(P_models.shape[0])
    ]
    returns = np.asarray([value[0] for value in values], dtype=float)
    costs = np.asarray([value[1] for value in values], dtype=float)
    return {
        "nominal_return": float(returns[0]),
        "worst_case_return": float(np.min(returns)),
        "nominal_cost": float(costs[0]),
        "worst_case_cost": float(np.max(costs)),
        "returns": returns,
        "costs": costs,
    }
