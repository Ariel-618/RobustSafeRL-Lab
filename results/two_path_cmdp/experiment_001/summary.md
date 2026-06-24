# Experiment 001: TwoPathCMDP robust policy comparison

## Configuration

- Discount factor: `0.95`
- Robust cost threshold: `0.1`
- Risky hazard probabilities: `(0.02, 0.2, 0.4)`
- Deterministic policies evaluated: `512`

## Results

| Policy | Start action | Nominal return | Worst-case return | Nominal cost | Worst-case cost | Robust feasible |
|---|---:|---:|---:|---:|---:|:---:|
| risky | 0 | 0.866761 | 0.324900 | 0.036689 | 0.596600 | False |
| safe | 1 | 0.814506 | 0.814506 | 0.000000 | 0.000000 | True |
| best_robust_feasible | 1 | 0.814506 | 0.814506 | 0.000000 | 0.000000 | True |

## Conclusion

The risky policy has the highest nominal return but violates the robust cost constraint. Exact enumeration selects start action 1, the safe path, as the best robust feasible deterministic policy.

Regenerate these files from the repository root with:

```powershell
py examples\02_solve_finite_rcmdp.py
```
