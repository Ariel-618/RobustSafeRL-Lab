"""A tiny finite-model constrained MDP with risky and safe paths."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class TwoPathCMDP:
    """Nine-state CMDP contrasting a short risky path with a longer safe path."""

    START = 0
    RISKY_1 = 1
    RISKY_2 = 2
    SAFE_1 = 3
    SAFE_2 = 4
    SAFE_3 = 5
    SAFE_4 = 6
    GOAL = 7
    HAZARD = 8

    def __init__(
        self,
        risky_hazard_probs: Tuple[float, ...] = (0.02, 0.20, 0.40),
    ) -> None:
        self.n_states = 9
        self.n_actions = 2
        self.risky_hazard_probs = tuple(float(p) for p in risky_hazard_probs)
        if not self.risky_hazard_probs:
            raise ValueError("risky_hazard_probs must contain at least one model.")
        if any(p < 0.0 or p > 1.0 for p in self.risky_hazard_probs):
            raise ValueError("Hazard probabilities must lie in [0, 1].")

        self.n_models = len(self.risky_hazard_probs)
        self._P_models, self._R_models, self._C_models = self._build_models()
        self._rng = np.random.default_rng()
        self._state = self.START
        self._model_index = 0

    def _build_models(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        shape = (self.n_models, self.n_states, self.n_actions, self.n_states)
        transitions = np.zeros(shape, dtype=float)

        for model_index, hazard_prob in enumerate(self.risky_hazard_probs):
            for action in range(self.n_actions):
                start_target = self.RISKY_1 if action == 0 else self.SAFE_1
                transitions[model_index, self.START, action, start_target] = 1.0

                transitions[
                    model_index, self.RISKY_1, action, self.RISKY_2
                ] = 1.0 - hazard_prob
                transitions[
                    model_index, self.RISKY_1, action, self.HAZARD
                ] = hazard_prob
                transitions[
                    model_index, self.RISKY_2, action, self.GOAL
                ] = 1.0 - hazard_prob
                transitions[
                    model_index, self.RISKY_2, action, self.HAZARD
                ] = hazard_prob

                transitions[model_index, self.SAFE_1, action, self.SAFE_2] = 1.0
                transitions[model_index, self.SAFE_2, action, self.SAFE_3] = 1.0
                transitions[model_index, self.SAFE_3, action, self.SAFE_4] = 1.0
                transitions[model_index, self.SAFE_4, action, self.GOAL] = 1.0

                transitions[model_index, self.GOAL, action, self.GOAL] = 1.0
                transitions[model_index, self.HAZARD, action, self.HAZARD] = 1.0

        rewards = transitions[..., self.GOAL].copy()
        costs = transitions[..., self.HAZARD].copy()
        rewards[:, self.GOAL, :] = 0.0
        costs[:, self.HAZARD, :] = 0.0
        return transitions, rewards, costs

    def transition_models(self) -> np.ndarray:
        """Return a copy of the finite transition-model array."""
        return self._P_models.copy()

    def reward_matrices(self) -> np.ndarray:
        """Return expected one-step rewards for every model."""
        return self._R_models.copy()

    def cost_matrices(self) -> np.ndarray:
        """Return expected one-step safety costs for every model."""
        return self._C_models.copy()

    def reset(
        self,
        seed: Optional[int] = None,
        model_index: Optional[int] = None,
    ) -> Tuple[int, dict[str, int]]:
        """Reset to Start and select the transition model for the episode."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if model_index is None:
            model_index = 0
        if not isinstance(model_index, (int, np.integer)):
            raise ValueError("model_index must be an integer.")
        if not 0 <= model_index < self.n_models:
            raise ValueError(f"model_index must be in [0, {self.n_models - 1}].")

        self._state = self.START
        self._model_index = int(model_index)
        return self._state, {"model_index": self._model_index}

    def step(
        self,
        action: int,
    ) -> Tuple[int, float, float, bool, bool, dict[str, int]]:
        """Sample one transition from the selected finite model."""
        if not isinstance(action, (int, np.integer)) or not 0 <= action < self.n_actions:
            raise ValueError(f"action must be an integer in [0, {self.n_actions - 1}].")

        previous_state = self._state
        probabilities = self._P_models[
            self._model_index, previous_state, int(action)
        ]
        next_state = int(self._rng.choice(self.n_states, p=probabilities))

        nonterminal = previous_state not in (self.GOAL, self.HAZARD)
        reward = float(next_state == self.GOAL and nonterminal)
        cost = float(next_state == self.HAZARD and nonterminal)
        self._state = next_state
        terminated = next_state in (self.GOAL, self.HAZARD)
        info = {"model_index": self._model_index}
        return next_state, reward, cost, terminated, False, info
