"""Run seeded episodes in the TwoPathCMDP benchmark."""

from robustsafe_rl.envs import TwoPathCMDP


def run_episode(env: TwoPathCMDP, start_action: int, model_index: int) -> None:
    state, info = env.reset(seed=7, model_index=model_index)
    total_reward = 0.0
    total_cost = 0.0
    trajectory = [state]

    while state not in (env.GOAL, env.HAZARD):
        action = start_action if state == env.START else 0
        state, reward, cost, terminated, truncated, info = env.step(action)
        trajectory.append(state)
        total_reward += reward
        total_cost += cost
        if terminated or truncated:
            break

    print(
        f"model={info['model_index']} start_action={start_action} "
        f"trajectory={trajectory} reward={total_reward:.1f} cost={total_cost:.1f}"
    )


def main() -> None:
    env = TwoPathCMDP()
    print("TwoPathCMDP: action 0 is risky; action 1 is safe.")
    run_episode(env, start_action=0, model_index=2)
    run_episode(env, start_action=1, model_index=2)


if __name__ == "__main__":
    main()
