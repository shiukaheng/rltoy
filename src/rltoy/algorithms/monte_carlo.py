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
    returns = [0.0] * len(transitions)  # Allocate one return-to-go for every transition.
    future_return = 0.0  # The return after the terminal transition is zero.
    for step in range(len(transitions) - 1, -1, -1):  # Work backward so each suffix return is already known.
        future_return = transitions[step][2] + gamma * future_return  # Add this reward to the discounted future return.
        returns[step] = future_return  # Save the return-to-go from this time step.

    seen = set()  # Remember state-action pairs encountered earlier in this episode.
    first_visits = []  # Collect only the return from each pair's first visit.
    for (state, action, _), return_ in zip(transitions, returns, strict=True):  # Scan the episode in chronological order.
        if (state, action) not in seen:
            seen.add((state, action))  # Mark this state-action pair as having its first visit.
            first_visits.append((state, action, return_))  # Keep its complete sampled return.
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

    rng = np.random.default_rng(seed)  # Create reproducible random choices for exploratory actions.
    states, actions = env.observation_space.n, env.action_space.n  # Read the finite table dimensions from the environment.
    q_values = np.zeros((states, actions))  # Start with no action-value estimates.
    return_sums = np.zeros_like(q_values)  # Accumulate sampled returns for each state-action pair.
    return_counts = np.zeros_like(q_values)  # Count first-visit return samples for each pair.
    episode_returns = np.zeros(config.episodes)  # Reserve one observed return for each episode.
    state_values = np.zeros((config.episodes, states))  # Record V(s) = max_a Q(s, a) after each episode.
    epsilons = decay_schedule(
        config.epsilon_initial,
        config.epsilon_final,
        config.epsilon_decay_ratio,
        config.episodes,
    )

    for episode, epsilon in enumerate(epsilons):  # Generate and learn from one complete episode at each exploration level.
        state, _ = env.reset(seed=seed if episode == 0 else None)  # Begin a new episode, seeding only the first reset.
        trajectory = [state]  # Track visited states for the optional observer.
        transitions = []  # Store the full episode before making Monte Carlo updates.

        while True:
            action = epsilon_greedy(q_values[state], epsilon, rng)  # Behave epsilon-greedily using current estimates.
            next_state, reward, terminated, truncated, _ = env.step(action)  # Sample one transition.
            transitions.append((state, action, reward))  # Keep it for the later complete-episode return calculation.
            trajectory.append(next_state)  # Extend the recorded state path.
            if terminated or truncated:
                break
            state = next_state  # Continue from the sampled successor state.

        for state, action, return_ in first_visit_returns(transitions, config.gamma):  # Update each pair from its first sampled visit.
            return_sums[state, action] += return_  # Add the complete discounted return sample.
            return_counts[state, action] += 1  # Count this return sample.
            q_values[state, action] = return_sums[state, action] / return_counts[state, action]  # Use the sample mean as Q(s, a).

        episode_returns[episode] = sum(reward for _, _, reward in transitions)  # Store the episode's observed return.
        state_values[episode] = q_values.max(axis=1)  # Derive a greedy state value from every Q row.
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
