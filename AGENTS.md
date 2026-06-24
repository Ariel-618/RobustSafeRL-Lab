# AGENTS.md

## Project Overview

RobustSafeRL-Lab is a lightweight benchmark and research lab for robust constrained reinforcement learning under model uncertainty.

The project focuses on robust constrained Markov decision processes (RCMDPs), especially small tabular environments, finite uncertainty sets, and clear evaluation protocols.

This is not intended to be a general-purpose RL training library like Tianshou, Stable-Baselines3, or RLlib. The main goal is to provide simple, readable, theoretically grounded RCMDP environments and benchmark tools.

## Core Research Question

Many reinforcement learning policies appear safe under nominal dynamics but may violate constraints under model uncertainty.

The central benchmark question is:

> Is a policy still safe when evaluated under worst-case transition dynamics?

## Current Scope

The current stage focuses on:

- Tabular finite-state finite-action CMDPs.
- Finite uncertainty sets over transition models.
- Deterministic policy evaluation.
- Brute-force deterministic policy enumeration for tiny RCMDPs.
- Clear examples and tests.

Do not implement deep RL, Gymnasium wrappers, PettingZoo, MARL, Docker, CI/CD, Hydra, WandB, or large-scale experiment systems unless explicitly requested.

## Mathematical Conventions

Use the following convention throughout the project:

- Rewards are better when larger.
- Costs are better when smaller.
- A policy is feasible if its worst-case discounted cost is below a threshold.

For a finite uncertainty set:

- `P_models` has shape `[n_models, n_states, n_actions, n_states]`.
- `R_models` has shape `[n_models, n_states, n_actions]`.
- `C_models` has shape `[n_models, n_states, n_actions]`.

For policy evaluation:

- A deterministic policy is a NumPy integer array of shape `[n_states]`.
- `policy[s]` is the action selected in state `s`.

The robust constrained objective is:

- Maximize worst-case reward return.
- Subject to worst-case cost being less than or equal to the cost threshold.

Specifically:

- `worst_case_return = min over models J_reward(policy, model)`
- `worst_case_cost = max over models J_cost(policy, model)`

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