# Experiment Results

This directory stores reproducible outputs from RobustSafeRL-Lab experiments.

Each experiment should use the following layout:

```text
results/
  <environment-or-benchmark>/
    experiment_<number>/
      summary.md
      metrics.csv
      comparison.png
```

- `summary.md` records the configuration, result table, and conclusion.
- `metrics.csv` stores machine-readable metrics for later analysis.
- `comparison.png` provides a visual comparison of the evaluated policies.

Generated results should identify the command needed to reproduce them.

## Available Experiments

- [`two_path_cmdp/experiment_001`](two_path_cmdp/experiment_001/summary.md):
  compares the risky policy, safe policy, and exact robust-feasible solution.
