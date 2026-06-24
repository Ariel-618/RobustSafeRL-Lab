"""Compare risky, safe, and robust-optimal deterministic policies."""

import numpy as np

from robustsafe_rl.algorithms.planning import solve_finite_rcmdp_by_enumeration
from robustsafe_rl.envs import TwoPathCMDP
from robustsafe_rl.evaluation import evaluate_policy_over_models


def print_result(
    name: str,
    policy: np.ndarray,
    evaluation: dict[str, object],
    cost_threshold: float,
) -> None:
    feasible = evaluation["worst_case_cost"] <= cost_threshold
    print(f"{name}:")
    print(f"  start action:       {policy[0]}")
    print(f"  nominal return:     {evaluation['nominal_return']:.6f}")
    print(f"  worst-case return:  {evaluation['worst_case_return']:.6f}")
    print(f"  nominal cost:       {evaluation['nominal_cost']:.6f}")
    print(f"  worst-case cost:    {evaluation['worst_case_cost']:.6f}")
    print(f"  robust feasible:    {feasible}")


def main() -> None:
    env = TwoPathCMDP()
    P_models = env.transition_models()
    R_models = env.reward_matrices()
    C_models = env.cost_matrices()
    gamma = 0.95
    cost_threshold = 0.10

    risky_policy = np.zeros(env.n_states, dtype=int)
    safe_policy = risky_policy.copy()
    safe_policy[env.START] = 1

    risky_evaluation = evaluate_policy_over_models(
        P_models, R_models, C_models, risky_policy, gamma
    )
    safe_evaluation = evaluate_policy_over_models(
        P_models, R_models, C_models, safe_policy, gamma
    )
    solution = solve_finite_rcmdp_by_enumeration(
        P_models,
        R_models,
        C_models,
        gamma,
        cost_threshold,
    )
    if solution is None:
        raise RuntimeError("No robust feasible policy was found.")

    print(f"gamma={gamma}, robust cost threshold={cost_threshold}\n")
    print_result("Risky policy", risky_policy, risky_evaluation, cost_threshold)
    print()
    print_result("Safe policy", safe_policy, safe_evaluation, cost_threshold)
    print()
    print_result(
        "Best robust feasible policy",
        solution["policy"],
        solution,
        cost_threshold,
    )
    print(f"  policies evaluated: {solution['policies_evaluated']}")


if __name__ == "__main__":
    main()
