"""First-visit Monte Carlo control, written to follow its textbook pseudocode."""

from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np

from rltoy.algorithms.tabular import (
    Observer,
    TrainingResult,
    TrainingSnapshot,
    decay_schedule,
    epsilon_greedy,
    observe_nothing,
)


@dataclass(frozen=True)
class MonteCarloConfig:
    episodes: int = 1_000
    gamma: float = 0.99
    epsilon_initial: float = 1.0
    epsilon_final: float = 0.1
    epsilon_decay_ratio: float = 0.9


def first_visit_returns(
    transitions: list[tuple[int, int, float]], gamma: float
) -> list[tuple[int, int, float]]:
    """Return the discounted return from each state-action pair's first episode visit."""
    returns = [0.0] * len(transitions)
    future_return = 0.0
    for step in range(len(transitions) - 1, -1, -1):
        future_return = transitions[step][2] + gamma * future_return
        returns[step] = future_return

    seen = set()
    first_visits = []
    for (state, action, _), return_ in zip(transitions, returns, strict=True):
        if (state, action) not in seen:
            seen.add((state, action))
            first_visits.append((state, action, return_))
    return first_visits


def train(
    env: gym.Env,
    config: MonteCarloConfig,
    seed: int | None = None,
    observer: Observer = observe_nothing,
) -> TrainingResult:
    """Estimate action values from complete episodes and improve epsilon-greedily."""
    if not isinstance(env.observation_space, gym.spaces.Discrete):
        raise ValueError("Monte Carlo control currently requires a discrete observation space")
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise ValueError("Monte Carlo control currently requires a discrete action space")

    rng = np.random.default_rng(seed)
    states, actions = env.observation_space.n, env.action_space.n
    q_values = np.zeros((states, actions))
    return_sums = np.zeros_like(q_values)
    return_counts = np.zeros_like(q_values)
    episode_returns = np.zeros(config.episodes)
    state_values = np.zeros((config.episodes, states))
    epsilons = decay_schedule(
        config.epsilon_initial,
        config.epsilon_final,
        config.epsilon_decay_ratio,
        config.episodes,
    )

    for episode, epsilon in enumerate(epsilons):
        state, _ = env.reset(seed=seed if episode == 0 else None)
        trajectory = [state]
        transitions = []

        while True:
            action = epsilon_greedy(q_values[state], epsilon, rng)
            next_state, reward, terminated, truncated, _ = env.step(action)
            transitions.append((state, action, reward))
            trajectory.append(next_state)
            if terminated or truncated:
                break
            state = next_state

        for state, action, return_ in first_visit_returns(transitions, config.gamma):
            return_sums[state, action] += return_
            return_counts[state, action] += 1
            q_values[state, action] = return_sums[state, action] / return_counts[state, action]

        episode_returns[episode] = sum(reward for _, _, reward in transitions)
        state_values[episode] = q_values.max(axis=1)
        observer(
            TrainingSnapshot(
                episode, trajectory[-1], tuple(trajectory), action_values=q_values
            )
        )

    return TrainingResult(q_values, episode_returns, state_values)


def main() -> None:
    import tyro

    from rltoy.cli.common import run

    @dataclass(frozen=True)
    class MonteCarloRunConfig(MonteCarloConfig):
        environment_id: str = "RLToy/BranchingRisk-v0"
        graph_path: Path | None = None
        seed: int | None = 0
        render: bool = False
        render_every_steps: int = 1
        render_delay_ms: int = 50

    run(train, tyro.cli(MonteCarloRunConfig), "First-visit Monte Carlo control")


if __name__ == "__main__":
    main()
