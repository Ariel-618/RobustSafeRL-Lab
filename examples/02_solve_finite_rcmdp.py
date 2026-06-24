"""Compare risky, safe, and robust-optimal deterministic policies."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from robustsafe_rl.algorithms.planning import solve_finite_rcmdp_by_enumeration
from robustsafe_rl.envs import TwoPathCMDP
from robustsafe_rl.evaluation import evaluate_policy_over_models

RESULTS_DIR = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "two_path_cmdp"
    / "experiment_001"
)


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


def build_result_rows(
    risky_policy: np.ndarray,
    safe_policy: np.ndarray,
    risky_evaluation: dict[str, object],
    safe_evaluation: dict[str, object],
    solution: dict[str, object],
    cost_threshold: float,
) -> list[dict[str, object]]:
    """Build a machine-readable comparison table."""
    policies = [
        ("risky", risky_policy, risky_evaluation),
        ("safe", safe_policy, safe_evaluation),
        ("best_robust_feasible", solution["policy"], solution),
    ]
    return [
        {
            "policy": name,
            "start_action": int(policy[0]),
            "nominal_return": float(evaluation["nominal_return"]),
            "worst_case_return": float(evaluation["worst_case_return"]),
            "nominal_cost": float(evaluation["nominal_cost"]),
            "worst_case_cost": float(evaluation["worst_case_cost"]),
            "cost_threshold": cost_threshold,
            "robust_feasible": bool(
                evaluation["worst_case_cost"] <= cost_threshold
            ),
        }
        for name, policy, evaluation in policies
    ]


def save_results(
    rows: list[dict[str, object]],
    gamma: float,
    cost_threshold: float,
    risky_hazard_probs: tuple[float, ...],
    policies_evaluated: int,
    output_dir: Path = RESULTS_DIR,
) -> None:
    """Save the experiment table, GitHub summary, and comparison figure."""
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    table_lines = [
        "| Policy | Start action | Nominal return | Worst-case return | "
        "Nominal cost | Worst-case cost | Robust feasible |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        table_lines.append(
            f"| {row['policy']} | {row['start_action']} | "
            f"{row['nominal_return']:.6f} | {row['worst_case_return']:.6f} | "
            f"{row['nominal_cost']:.6f} | {row['worst_case_cost']:.6f} | "
            f"{row['robust_feasible']} |"
        )

    summary = "\n".join(
        [
            "# Experiment 001: TwoPathCMDP robust policy comparison",
            "",
            "## Configuration",
            "",
            f"- Discount factor: `{gamma}`",
            f"- Robust cost threshold: `{cost_threshold}`",
            f"- Risky hazard probabilities: `{risky_hazard_probs}`",
            f"- Deterministic policies evaluated: `{policies_evaluated}`",
            "",
            "## Results",
            "",
            *table_lines,
            "",
            "## Conclusion",
            "",
            "The risky policy has the highest nominal return but violates the "
            "robust cost constraint. Exact enumeration selects start action 1, "
            "the safe path, as the best robust feasible deterministic policy.",
            "",
            "Regenerate these files from the repository root with:",
            "",
            "```powershell",
            "py examples\\02_solve_finite_rcmdp.py",
            "```",
            "",
        ]
    )
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")

    labels = [str(row["policy"]).replace("_", "\n") for row in rows]
    x_positions = np.arange(len(rows))
    width = 0.36
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].bar(
        x_positions - width / 2,
        [row["nominal_return"] for row in rows],
        width,
        label="Nominal",
        color="#4C78A8",
    )
    axes[0].bar(
        x_positions + width / 2,
        [row["worst_case_return"] for row in rows],
        width,
        label="Worst case",
        color="#F58518",
    )
    axes[0].set_title("Discounted reward return")
    axes[0].set_xticks(x_positions, labels)
    axes[0].set_ylabel("Return")
    axes[0].legend()

    axes[1].bar(
        x_positions - width / 2,
        [row["nominal_cost"] for row in rows],
        width,
        label="Nominal",
        color="#54A24B",
    )
    axes[1].bar(
        x_positions + width / 2,
        [row["worst_case_cost"] for row in rows],
        width,
        label="Worst case",
        color="#E45756",
    )
    axes[1].axhline(
        cost_threshold,
        color="#B279A2",
        linestyle="--",
        label="Cost threshold",
    )
    axes[1].set_title("Discounted safety cost")
    axes[1].set_xticks(x_positions, labels)
    axes[1].set_ylabel("Cost")
    axes[1].legend()

    figure.suptitle("TwoPathCMDP: nominal versus robust evaluation")
    figure.tight_layout()
    figure.savefig(output_dir / "comparison.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


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

    rows = build_result_rows(
        risky_policy,
        safe_policy,
        risky_evaluation,
        safe_evaluation,
        solution,
        cost_threshold,
    )
    save_results(
        rows,
        gamma,
        cost_threshold,
        env.risky_hazard_probs,
        int(solution["policies_evaluated"]),
    )

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
    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
