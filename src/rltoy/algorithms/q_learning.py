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
    rng = np.random.default_rng(seed)  # Create reproducible random choices for exploratory actions.
    q_values = np.zeros((env.observation_space.n, env.action_space.n))  # Start with no action-value estimates.
    returns = np.zeros(config.episodes)  # Reserve one undiscounted return for each episode.
    state_values = np.zeros((config.episodes, env.observation_space.n))  # Record V(s) = max_a Q(s, a) after each episode.
    alphas = decay_schedule(config.alpha_initial, config.alpha_final, config.alpha_decay_ratio, config.episodes)  # Schedule the TD step size.
    epsilons = decay_schedule(config.epsilon_initial, config.epsilon_final, config.epsilon_decay_ratio, config.episodes)  # Schedule exploration.

    for episode, (alpha, epsilon) in enumerate(zip(alphas, epsilons, strict=True)):  # Train once with each pair of schedule values.
        state, _ = env.reset(seed=seed if episode == 0 else None)  # Begin a new episode, seeding only the first reset.
        trajectory = [state]  # Track visited states for the optional observer.
        episode_return = 0.0  # Accumulate this episode's actual rewards.

        while True:
            action = epsilon_greedy(q_values[state], epsilon, rng)  # Behave epsilon-greedily using the current Q table.
            next_state, reward, terminated, truncated, _ = env.step(action)  # Sample the environment transition.
            done = terminated or truncated  # Treat natural endings and time limits as episode boundaries.

            target = reward if done else reward + config.gamma * q_values[next_state].max()  # Bootstrap from the greedy next-state value, not the next behavior action.
            q_values[state, action] += alpha * (target - q_values[state, action])  # Move Q(s, a) toward the off-policy TD target.

            episode_return += reward  # Add the transition reward to the episode total.
            trajectory.append(next_state)  # Extend the recorded state path.
            observer(
                TrainingSnapshot(
                    episode, next_state, tuple(trajectory), action_values=q_values
                )
            )
            if done:
                break
            state = next_state  # Continue from the sampled successor state.

        returns[episode] = episode_return  # Store this episode's observed return.
        state_values[episode] = q_values.max(axis=1)  # Derive a greedy state value from every Q row.

    return TrainingResult(q_values, returns, state_values)


def main() -> None:
    import tyro

    from rltoy.cli.common import RunConfig, run

    run(train, tyro.cli(RunConfig), "Q-learning")


if __name__ == "__main__":
    main()
