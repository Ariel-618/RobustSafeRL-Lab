# RobustSafeRL-Lab

Minimal benchmarks and algorithms for robust constrained reinforcement learning under model uncertainty.

## Overview

Many reinforcement learning policies appear safe under nominal dynamics but may violate constraints when the transition model changes. RobustSafeRL-Lab aims to provide lightweight environments, algorithms, and evaluation tools for studying robust constrained Markov decision processes.

The initial focus is on tabular robust constrained MDPs, including simple chain and grid-world environments.

## Project Goals

- Provide minimal environments for robust constrained reinforcement learning.
- Separate nominal performance from worst-case performance.
- Separate reward optimization from constraint satisfaction.
- Implement simple and readable tabular algorithms.
- Build a foundation for future robust safe RL and robust safe MARL benchmarks.

## Current Status

This project is under early development.

The first milestone is:

- Chain CMDP environment
- Nominal policy evaluation
- Worst-case policy evaluation
- Value iteration
- Robust value iteration
- Basic benchmark table

## Installation

```bash
git clone https://github.com/Ariel-618/RobustSafeRL-Lab.git
cd RobustSafeRL-Lab
pip install -e .