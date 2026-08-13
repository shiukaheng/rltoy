"""Tabular SARSA, written to follow its textbook pseudocode."""

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
    """Learn an action-value table with on-policy one-step TD control."""
    rng = np.random.default_rng(seed)  # Create reproducible random choices for exploratory actions.
    q_values = np.zeros((env.observation_space.n, env.action_space.n))  # Start with no action-value estimates.
    returns = np.zeros(config.episodes)  # Reserve one undiscounted return for each episode.
    state_values = np.zeros((config.episodes, env.observation_space.n))  # Record V(s) = max_a Q(s, a) after each episode.
    alphas = decay_schedule(config.alpha_initial, config.alpha_final, config.alpha_decay_ratio, config.episodes)  # Schedule the TD step size.
    epsilons = decay_schedule(config.epsilon_initial, config.epsilon_final, config.epsilon_decay_ratio, config.episodes)  # Schedule exploration.

    for episode, (alpha, epsilon) in enumerate(zip(alphas, epsilons, strict=True)):  # Train once with each pair of schedule values.
        state, _ = env.reset(seed=seed if episode == 0 else None)  # Begin a new episode, seeding only the first reset.
        action = epsilon_greedy(q_values[state], epsilon, rng)  # Choose the first action from the behavior policy.
        trajectory = [state]  # Track visited states for the optional observer.
        episode_return = 0.0  # Accumulate this episode's actual rewards.

        while True:
            next_state, reward, terminated, truncated, _ = env.step(action)  # Sample the transition for the already selected action.
            done = terminated or truncated  # Treat natural endings and time limits as episode boundaries.
            if done:
                target = reward  # A terminal transition has no successor value to bootstrap from.
            else:
                next_action = epsilon_greedy(q_values[next_state], epsilon, rng)  # Sample the next action from the same behavior policy.
                target = reward + config.gamma * q_values[next_state, next_action]  # Bootstrap from that sampled on-policy action.
            q_values[state, action] += alpha * (target - q_values[state, action])  # Move Q(s, a) toward the SARSA target.

            episode_return += reward  # Add the transition reward to the episode total.
            trajectory.append(next_state)  # Extend the recorded state path.
            observer(
                TrainingSnapshot(
                    episode, next_state, tuple(trajectory), action_values=q_values
                )
            )
            if done:
                break
            state, action = next_state, next_action  # Reuse the sampled action as the next SARSA pair.

        returns[episode] = episode_return  # Store this episode's observed return.
        state_values[episode] = q_values.max(axis=1)  # Derive a greedy state value from every Q row.

    return TrainingResult(q_values, returns, state_values)


def main() -> None:
    import tyro

    from rltoy.cli.common import RunConfig, run

    run(train, tyro.cli(RunConfig), "SARSA")


if __name__ == "__main__":
    main()
