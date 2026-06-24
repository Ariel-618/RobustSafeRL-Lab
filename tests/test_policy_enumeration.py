import numpy as np

from robustsafe_rl.algorithms.planning import (
    enumerate_deterministic_policies,
    solve_finite_rcmdp_by_enumeration,
)
from robustsafe_rl.envs import TwoPathCMDP


def test_enumerates_every_deterministic_policy_in_lexicographic_order():
    policies = list(enumerate_deterministic_policies(9, 2))

    assert len(policies) == 512
    np.testing.assert_array_equal(policies[0], np.zeros(9, dtype=int))
    np.testing.assert_array_equal(policies[-1], np.ones(9, dtype=int))


def test_solver_selects_a_robust_feasible_safe_policy():
    env = TwoPathCMDP()
    threshold = 0.10
    solution = solve_finite_rcmdp_by_enumeration(
        env.transition_models(),
        env.reward_matrices(),
        env.cost_matrices(),
        gamma=0.95,
        cost_threshold=threshold,
    )

    assert solution is not None
    assert solution["feasible"]
    assert solution["worst_case_cost"] <= threshold
    assert solution["policy"][env.START] == 1
    assert solution["policies_evaluated"] == 512
