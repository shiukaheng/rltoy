"""Tabular Q-learning, written to follow its textbook pseudocode."""

import gymnasium as gym
import numpy as np

from rltoy.algorithms.tabular import (
    Observer,
    TabularConfig,
    TrainingResult,
    TrainingSnapshot,
    decay_schedule,
    epsilon_greedy,
    observe_nothing,
)


def train(
    env: gym.Env,
    config: TabularConfig,
    seed: int | None = None,
    observer: Observer = observe_nothing,
) -> TrainingResult:
    """Learn an action-value table with off-policy TD control."""
    rng = np.random.default_rng(seed)
    q_values = np.zeros((env.observation_space.n, env.action_space.n))
    returns = np.zeros(config.episodes)
    state_values = np.zeros((config.episodes, env.observation_space.n))
    alphas = decay_schedule(config.alpha_initial, config.alpha_final, config.alpha_decay_ratio, config.episodes)
    epsilons = decay_schedule(config.epsilon_initial, config.epsilon_final, config.epsilon_decay_ratio, config.episodes)

    for episode, (alpha, epsilon) in enumerate(zip(alphas, epsilons, strict=True)):
        state, _ = env.reset(seed=seed if episode == 0 else None)
        trajectory = [state]
        episode_return = 0.0

        while True:
            action = epsilon_greedy(q_values[state], epsilon, rng)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            target = reward if done else reward + config.gamma * q_values[next_state].max()
            q_values[state, action] += alpha * (target - q_values[state, action])

            episode_return += reward
            trajectory.append(next_state)
            observer(
                TrainingSnapshot(
                    episode, next_state, tuple(trajectory), action_values=q_values
                )
            )
            if done:
                break
            state = next_state

        returns[episode] = episode_return
        state_values[episode] = q_values.max(axis=1)

    return TrainingResult(q_values, returns, state_values)


def main() -> None:
    import tyro

    from rltoy.cli.common import RunConfig, run

    run(train, tyro.cli(RunConfig), "Q-learning")


if __name__ == "__main__":
    main()
