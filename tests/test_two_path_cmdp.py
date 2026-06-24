import numpy as np

from robustsafe_rl.envs import TwoPathCMDP


def test_model_shapes_and_stochastic_transitions():
    env = TwoPathCMDP()
    P_models = env.transition_models()

    assert P_models.shape == (3, 9, 2, 9)
    assert env.reward_matrices().shape == (3, 9, 2)
    assert env.cost_matrices().shape == (3, 9, 2)
    np.testing.assert_allclose(P_models.sum(axis=-1), 1.0)


def test_start_actions_choose_the_two_paths():
    env = TwoPathCMDP()
    P_models = env.transition_models()

    np.testing.assert_allclose(P_models[:, env.START, 0, env.RISKY_1], 1.0)
    np.testing.assert_allclose(P_models[:, env.START, 1, env.SAFE_1], 1.0)


def test_risky_probabilities_apply_to_both_risky_legs():
    env = TwoPathCMDP()
    P_models = env.transition_models()
    expected = np.asarray(env.risky_hazard_probs)

    for state in (env.RISKY_1, env.RISKY_2):
        for action in range(env.n_actions):
            np.testing.assert_allclose(
                P_models[:, state, action, env.HAZARD], expected
            )


def test_safe_path_has_no_hazard_probability():
    env = TwoPathCMDP()
    P_models = env.transition_models()
    safe_states = [env.SAFE_1, env.SAFE_2, env.SAFE_3, env.SAFE_4]

    np.testing.assert_allclose(P_models[:, safe_states, :, env.HAZARD], 0.0)


def test_reset_and_step_use_selected_model():
    env = TwoPathCMDP()
    state, info = env.reset(seed=1, model_index=2)
    assert state == env.START
    assert info == {"model_index": 2}

    state, reward, cost, terminated, truncated, info = env.step(1)
    assert state == env.SAFE_1
    assert reward == 0.0
    assert cost == 0.0
    assert not terminated
    assert not truncated
    assert info == {"model_index": 2}
