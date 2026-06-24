# AGENTS.md

## Project Overview

RobustSafeRL-Lab is a lightweight benchmark and research lab for robust
constrained reinforcement learning under model uncertainty.

The project studies robust constrained Markov decision processes (RCMDPs) with
an emphasis on:

- Small, interpretable single-agent environments.
- Explicit uncertainty sets over transition dynamics.
- Transparent planning and optimization algorithms.
- Reproducible comparisons between nominal and robust safety.
- Implementations that remain close to the underlying mathematics.

This is not intended to become a general-purpose reinforcement-learning
framework such as Tianshou, Stable-Baselines3, or RLlib. Prefer a small,
readable research codebase over broad framework integration.

## Core Research Questions

The central benchmark question is:

> Is a policy still safe when evaluated under worst-case transition dynamics?

Related questions include:

- How much nominal reward must be sacrificed for robust feasibility?
- How does the choice and size of an uncertainty set change the optimal policy?
- When do nominally safe policies violate constraints under model uncertainty?
- How conservative are different robust constrained algorithms?
- How closely do scalable algorithms match exact solutions on small problems?

## Current Baseline

The first prototype is complete and should remain a compact correctness
baseline:

- `TwoPathCMDP`, with a short risky route and a longer safe route.
- A finite set of complete transition models.
- Exact discounted evaluation of deterministic policies.
- Brute-force deterministic policy enumeration for tiny finite-model RCMDPs.
- Reproducible result tables and figures under `results/`.

Treat this implementation as an exact oracle for small test problems, not as a
scalable solution method or the final scope of the project.

## Current Research Priorities

The next stage should broaden the benchmark along three coordinated axes.

### 1. Representative uncertainty sets

Prioritize uncertainty in transition dynamics. Add uncertainty families
incrementally and keep their semantics explicit:

- Finite sets of complete transition models.
- State-action rectangular uncertainty sets.
- L1 or total-variation ambiguity sets.
- KL or other divergence-based ambiguity sets.
- Wasserstein ambiguity sets when the simpler formulations are stable.

Do not silently treat a finite set of globally coupled models as equivalent to
a rectangular uncertainty set. State whether nature selects one complete model
for the full trajectory or may select transitions independently across
state-action pairs.

### 2. Representative single-agent algorithms

Build a small but meaningful set of baselines rather than many superficial
implementations:

- Exact enumeration for tiny deterministic-policy problems.
- Nominal CMDP baselines.
- Robust policy evaluation.
- Robust value or policy iteration where rectangularity permits dynamic
  programming.
- Occupancy-measure or linear-programming formulations where appropriate.
- Lagrangian and primal-dual robust constrained methods.

Support randomized policies when required by the mathematical formulation.
Clearly distinguish exact algorithms, approximation algorithms, and learning
algorithms. Validate scalable methods against exact solutions whenever the
problem is small enough.

### 3. Representative environments

Add environments because they expose a distinct robustness or safety issue,
not merely to increase the environment count. Strong candidates include:

- Robust gridworld navigation with uncertain slipping or hazards.
- Machine replacement with uncertain deterioration.
- Inventory control with uncertain demand.
- Queueing or admission control with uncertain arrival/service dynamics.

Each environment should have an interpretable reward-safety tradeoff, a clearly
documented uncertainty model, and at least one small configuration that can be
solved exactly.

## Scope Boundaries

The near-term scope is:

- Single-agent CMDPs and RCMDPs.
- Finite state and action spaces.
- Discounted infinite-horizon or clearly specified episodic objectives.
- Model-based planning and small-scale learning baselines.
- NumPy-first implementations with minimal dependencies.

Do not add deep RL, continuous-control suites, Gymnasium wrappers, PettingZoo,
MARL, Docker, CI/CD, Hydra, WandB, distributed training, or large experiment
platforms unless explicitly requested.

## Mathematical Conventions

Use the following conventions throughout the project:

- Rewards are better when larger.
- Costs are better when smaller.
- A policy is robust feasible when its worst-case discounted cost is less than
  or equal to the cost threshold.
- `gamma` satisfies `0 <= gamma < 1`.
- The nominal model or nominal transition kernel must be identified explicitly.

For a finite uncertainty set:

- `P_models` has shape `[n_models, n_states, n_actions, n_states]`.
- `R_models` has shape `[n_models, n_states, n_actions]`.
- `C_models` has shape `[n_models, n_states, n_actions]`.
- Model index `0` is nominal unless an API explicitly states otherwise.

For deterministic policies:

- A policy is a NumPy integer array of shape `[n_states]`.
- `policy[s]` is the action selected in state `s`.

For randomized stationary policies:

- A policy should use shape `[n_states, n_actions]`.
- Each row is a probability distribution over actions.

For finite complete-model uncertainty:

- `worst_case_return = min_m J_reward(policy, model_m)`
- `worst_case_cost = max_m J_cost(policy, model_m)`

The robust constrained objective is:

- Maximize worst-case reward return.
- Subject to worst-case cost being less than or equal to the threshold.

Do not apply these finite-model formulas unchanged to rectangular or
distributional uncertainty sets; use the worst-case operator appropriate to
the stated uncertainty semantics.

## API and Implementation Principles

- Keep public array shapes, return values, and uncertainty semantics documented.
- Validate dimensions, probability normalization, action ranges, and discount
  factors at public boundaries.
- Prefer exact linear algebra or optimization for tabular problems.
- Keep environment construction separate from policy evaluation and solvers.
- Keep uncertainty-set logic separate from environment-specific code whenever
  it is reusable.
- Avoid hidden global state and implicit randomness; expose seeds where
  simulation is stochastic.
- Preserve compatibility with the Python version declared in `pyproject.toml`.
- Keep dependencies minimal and justify each new dependency.

## Evaluation and Reporting

At minimum, comparable experiments should report:

- Nominal return.
- Worst-case return.
- Nominal cost.
- Worst-case cost.
- Robust feasibility.
- Constraint violation, when infeasible.
- Runtime and iteration count for iterative algorithms.
- The uncertainty-set type and its parameters.
- The cost threshold, discount factor, and policy class.

When meaningful, also report robustness gaps and sensitivity curves over the
uncertainty radius or cost threshold.

Store reproducible outputs under:

```text
results/
  <environment-or-benchmark>/
    experiment_<number>/
      summary.md
      metrics.csv
      comparison.png
```

Every saved experiment should record enough configuration and the exact command
needed to regenerate it. Avoid committing opaque binary artifacts when a small
text or CSV representation is sufficient.

## Testing Expectations

Every new environment, uncertainty set, evaluator, or solver should include
focused tests.

Tests should cover, as applicable:

- Array shapes and probability normalization.
- Terminal and boundary behavior.
- Known analytical values on tiny problems.
- Nominal versus worst-case aggregation.
- Feasibility and constraint violations.
- Agreement with enumeration or another exact oracle.
- Reproducibility under fixed seeds.
- Failure on invalid inputs.

Run the complete test suite before handing off changes:

```bash
py -m pytest
```

Use `python -m pytest` if the Python launcher is unavailable.

## Package Structure

Use the following structure:

```text
robustsafe_rl/
  core/
  envs/
  uncertainty/
  algorithms/
    planning/
  evaluation/
  visualization/
  utils/

examples/
tests/
docs/
results/
```

Add new modules only when they have a clear responsibility. Prefer a small
number of cohesive modules over premature abstraction.
