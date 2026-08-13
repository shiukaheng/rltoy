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
    rng = np.random.default_rng(seed)  # Create reproducible random choices for exploratory actions.
    q_values = np.zeros((env.observation_space.n, env.action_space.n))  # Start with no action-value estimates.
    returns = np.zeros(config.episodes)  # Reserve one undiscounted return for each episode.
    state_values = np.zeros((config.episodes, env.observation_space.n))  # Record V(s) = max_a Q(s, a) after each episode.
    alphas = decay_schedule(config.alpha_initial, config.alpha_final, config.alpha_decay_ratio, config.episodes)  # Schedule the TD step size.
    epsilons = decay_schedule(config.epsilon_initial, config.epsilon_final, config.epsilon_decay_ratio, config.episodes)  # Schedule exploration.

    for episode, (alpha, epsilon) in enumerate(zip(alphas, epsilons, strict=True)):  # Train once with each pair of schedule values.
        traces = np.zeros_like(q_values)  # Reset all accumulating eligibility traces for this episode.
        state, _ = env.reset(seed=seed if episode == 0 else None)  # Begin a new episode, seeding only the first reset.
        action = epsilon_greedy(q_values[state], epsilon, rng)  # Choose the first action from the behavior policy.
        trajectory = [state]  # Track visited states for the optional observer.
        episode_return = 0.0  # Accumulate this episode's actual rewards.

        while True:
            next_state, reward, terminated, truncated, _ = env.step(action)  # Sample the transition for the selected action.
            done = terminated or truncated  # Treat natural endings and time limits as episode boundaries.
            if done:
                target = reward  # A terminal transition has no successor value to bootstrap from.
            else:
                next_action = epsilon_greedy(q_values[next_state], epsilon, rng)  # Sample the next action from the behavior policy.
                target = reward + config.gamma * q_values[next_state, next_action]  # Form the on-policy one-step target.
            td_error = target - q_values[state, action]  # Measure this transition's one-step prediction error.
            traces[state, action] += 1.0  # Increase eligibility for the state-action pair just visited.
            q_values += alpha * td_error * traces  # Apply the error to every pair in proportion to its trace.
            traces *= config.gamma * config.trace_decay  # Decay past eligibilities through time.

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
    from dataclasses import dataclass

    import tyro

    from rltoy.cli.common import RunConfig, run

    @dataclass(frozen=True)
    class Config(RunConfig):
        trace_decay: float = 0.8

    run(train, tyro.cli(Config), "SARSA(lambda)")


if __name__ == "__main__":
    main()
