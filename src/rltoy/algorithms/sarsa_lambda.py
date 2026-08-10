"""Tabular accumulating-trace SARSA(lambda), kept as visible pseudocode."""

from dataclasses import dataclass

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


@dataclass(frozen=True)
class SarsaLambdaConfig(TabularConfig):
    trace_decay: float = 0.8


def train(
    env: gym.Env,
    config: SarsaLambdaConfig,
    seed: int | None = None,
    observer: Observer = observe_nothing,
) -> TrainingResult:
    """Learn action values with on-policy TD control and accumulating traces."""
    rng = np.random.default_rng(seed)
    q_values = np.zeros((env.observation_space.n, env.action_space.n))
    returns = np.zeros(config.episodes)
    state_values = np.zeros((config.episodes, env.observation_space.n))
    alphas = decay_schedule(config.alpha_initial, config.alpha_final, config.alpha_decay_ratio, config.episodes)
    epsilons = decay_schedule(config.epsilon_initial, config.epsilon_final, config.epsilon_decay_ratio, config.episodes)

    for episode, (alpha, epsilon) in enumerate(zip(alphas, epsilons, strict=True)):
        traces = np.zeros_like(q_values)
        state, _ = env.reset(seed=seed if episode == 0 else None)
        action = epsilon_greedy(q_values[state], epsilon, rng)
        trajectory = [state]
        episode_return = 0.0

        while True:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if done:
                target = reward
            else:
                next_action = epsilon_greedy(q_values[next_state], epsilon, rng)
                target = reward + config.gamma * q_values[next_state, next_action]
            td_error = target - q_values[state, action]
            traces[state, action] += 1.0
            q_values += alpha * td_error * traces
            traces *= config.gamma * config.trace_decay

            episode_return += reward
            trajectory.append(next_state)
            observer(
                TrainingSnapshot(
                    episode, next_state, tuple(trajectory), action_values=q_values
                )
            )
            if done:
                break
            state, action = next_state, next_action

        returns[episode] = episode_return
        state_values[episode] = q_values.max(axis=1)

    return TrainingResult(q_values, returns, state_values)


def main() -> None:
    from dataclasses import dataclass

    import tyro

    from rltoy.cli.common import RunConfig, run

    @dataclass(frozen=True)
    class Config(RunConfig):
        trace_decay: float = 0.8

    run(train, tyro.cli(Config), "SARSA(lambda)")


if __name__ == "__main__":
    main()
