import numpy as np

from robustsafe_rl.envs import TwoPathCMDP
from robustsafe_rl.evaluation import (
    evaluate_deterministic_policy,
    evaluate_policy_over_models,
)


def test_single_model_policy_evaluation_matches_model_aggregation():
    env = TwoPathCMDP()
    policy = np.zeros(env.n_states, dtype=int)
    P_models = env.transition_models()
    R_models = env.reward_matrices()
    C_models = env.cost_matrices()

    reward, cost = evaluate_deterministic_policy(
        P_models[0], R_models[0], C_models[0], policy, gamma=0.95
    )
    evaluation = evaluate_policy_over_models(
        P_models, R_models, C_models, policy, gamma=0.95
    )

    assert reward == evaluation["nominal_return"]
    assert cost == evaluation["nominal_cost"]
    assert evaluation["returns"].shape == (env.n_models,)
    assert evaluation["costs"].shape == (env.n_models,)
    assert evaluation["worst_case_return"] == np.min(evaluation["returns"])
    assert evaluation["worst_case_cost"] == np.max(evaluation["costs"])


def test_risky_policy_trades_nominal_return_for_worst_case_cost():
    env = TwoPathCMDP()
    risky_policy = np.zeros(env.n_states, dtype=int)
    safe_policy = risky_policy.copy()
    safe_policy[env.START] = 1
    inputs = (
        env.transition_models(),
        env.reward_matrices(),
        env.cost_matrices(),
    )

    risky = evaluate_policy_over_models(*inputs, risky_policy, gamma=0.95)
    safe = evaluate_policy_over_models(*inputs, safe_policy, gamma=0.95)

    assert risky["nominal_return"] > safe["nominal_return"]
    assert risky["worst_case_cost"] > safe["worst_case_cost"]
    assert safe["worst_case_cost"] == 0.0
