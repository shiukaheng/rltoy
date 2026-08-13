"""Generalized Advantage Estimation (GAE) with separate policy and value networks."""

from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from rltoy.algorithms.tabular import Observer, TrainingResult, TrainingSnapshot, observe_nothing


@dataclass(frozen=True)
class GAEConfig:
    episodes: int = 1_000
    gamma: float = 0.99
    gae_lambda: float = 0.95
    hidden_units: int = 32
    policy_learning_rate: float = 1e-3
    value_learning_rate: float = 1e-3
    device: str = "cpu"


class PolicyNetwork(nn.Module):
    def __init__(self, states: int, actions: int, hidden_units: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(states, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, actions),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.layers(states)


class ValueNetwork(nn.Module):
    def __init__(self, states: int, hidden_units: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(states, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.layers(states).squeeze(1)


def one_hot(states: np.ndarray, state_count: int, device: torch.device) -> torch.Tensor:
    state_ids = torch.as_tensor(states, dtype=torch.long, device=device)
    return torch.nn.functional.one_hot(state_ids, state_count).to(torch.float32)


def generalized_advantages(
    rewards: list[float],
    values: np.ndarray,
    bootstrap_value: float,
    gamma: float,
    gae_lambda: float,
    terminated: bool,
) -> np.ndarray:
    """Compute GAE from one completed trajectory and an optional bootstrap value."""
    advantages = np.zeros(len(rewards))  # Allocate one advantage estimate for every trajectory step.
    future_advantage = 0.0  # There is no advantage beyond the trajectory end.
    for step in range(len(rewards) - 1, -1, -1):  # Apply the GAE recursion from the end of the trajectory.
        is_final_step = step == len(rewards) - 1  # Identify the boundary where a bootstrap may be needed.
        next_value = bootstrap_value if is_final_step else values[step + 1]  # Use V(s_{t+1}) from the trajectory or its final bootstrap.
        bootstrap = 0.0 if is_final_step and terminated else 1.0  # Do not bootstrap beyond a true terminal state.
        continuation = 0.0 if is_final_step else 1.0  # Stop the advantage recursion at the trajectory boundary.
        td_error = rewards[step] + gamma * bootstrap * next_value - values[step]  # Compute the one-step TD residual.
        future_advantage = td_error + gamma * gae_lambda * continuation * future_advantage  # Mix this residual with later residuals.
        advantages[step] = future_advantage  # Save A^GAE_t for the current action.
    return advantages


def train(
    env: gym.Env,
    config: GAEConfig,
    seed: int | None = None,
    observer: Observer = observe_nothing,
) -> TrainingResult:
    """Learn from trajectory GAE advantages using a learned state-value baseline."""
    if not 0.0 <= config.gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be between 0 and 1")
    if not isinstance(env.observation_space, gym.spaces.Discrete):
        raise ValueError("GAE currently requires a discrete observation space")
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise ValueError("GAE currently requires a discrete action space")

    if seed is not None:
        torch.manual_seed(seed)  # Make network initialization and action samples reproducible.
    device = torch.device(config.device)  # Place network computations on the requested device.
    states, actions = env.observation_space.n, env.action_space.n  # Read the one-hot input and action-output dimensions.
    policy = PolicyNetwork(states, actions, config.hidden_units).to(device)  # Create the stochastic actor.
    value = ValueNetwork(states, config.hidden_units).to(device)  # Create the state-value baseline.
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=config.policy_learning_rate)  # Optimize the actor separately.
    value_optimizer = torch.optim.Adam(value.parameters(), lr=config.value_learning_rate)  # Optimize the baseline separately.
    episode_returns = np.zeros(config.episodes)  # Reserve one observed return for each episode.
    value_history = np.zeros((config.episodes, states))  # Record the critic's estimate for every state.

    for episode in range(config.episodes):  # Collect a trajectory, then update both networks once.
        state, _ = env.reset(seed=seed if episode == 0 else None)  # Begin a new episode, seeding only the first reset.
        trajectory = [state]  # Track visited states for the optional observer.
        visited_states = []  # Keep states aligned with sampled actions and rewards.
        rewards = []  # Keep rewards for the later GAE calculation.
        log_probabilities = []  # Keep differentiable log probabilities of sampled actions.

        while True:
            logits = policy(one_hot(np.asarray([state]), states, device))[0]  # Produce unnormalized action preferences for this state.
            distribution = torch.distributions.Categorical(logits=logits)  # Turn preferences into a categorical policy.
            action = distribution.sample()  # Sample an action from the current policy.
            next_state, reward, terminated, truncated, _ = env.step(action.item())  # Sample the environment transition.
            visited_states.append(state)  # Retain the state whose action received this reward.
            rewards.append(reward)  # Retain the reward for the GAE recursion.
            log_probabilities.append(distribution.log_prob(action))  # Retain the score-function term for this action.
            trajectory.append(next_state)  # Extend the recorded state path.
            if terminated or truncated:
                break
            state = next_state  # Continue from the sampled successor state.

        with torch.no_grad():  # Treat values used to construct targets as fixed estimates.
            visited = one_hot(np.asarray(visited_states), states, device)  # Encode the trajectory states in one batch.
            old_values = value(visited).cpu().numpy()  # Estimate V(s_t) for every trajectory step.
            bootstrap_value = 0.0 if terminated else value(
                one_hot(np.asarray([next_state]), states, device)
            ).item()  # Bootstrap a truncation from V(s_T), but not a true terminal state.
        advantages = generalized_advantages(
            rewards, old_values, bootstrap_value, config.gamma, config.gae_lambda, terminated
        )  # Combine TD residuals into the trajectory's GAE advantages.
        returns_to_go = advantages + old_values  # Convert advantages into value-regression targets.
        advantages_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=device)  # Move fixed advantages into the actor loss.
        returns_tensor = torch.as_tensor(returns_to_go, dtype=torch.float32, device=device)  # Move fixed return targets into the critic loss.
        policy_loss = -(torch.stack(log_probabilities) * advantages_tensor).sum()  # Increase action likelihood in proportion to GAE advantage.
        value_loss = torch.nn.functional.mse_loss(value(visited), returns_tensor)  # Train the critic to fit GAE return targets.

        policy_optimizer.zero_grad()  # Clear actor gradients from the previous trajectory.
        policy_loss.backward()  # Differentiate the GAE policy-gradient objective.
        policy_optimizer.step()  # Update the policy.
        value_optimizer.zero_grad()  # Clear critic gradients from the previous trajectory.
        value_loss.backward()  # Differentiate the value regression loss.
        value_optimizer.step()  # Update the state-value baseline.

        with torch.no_grad():  # Reporting values and probabilities does not need an autograd graph.
            all_states = one_hot(np.arange(states), states, device)  # Encode every state for a complete snapshot.
            probabilities = torch.softmax(policy(all_states), dim=1).cpu().numpy()  # Evaluate pi(a | s) for every state.
            estimated_values = value(all_states).cpu().numpy()  # Evaluate V(s) for every state.
        episode_returns[episode] = sum(rewards)  # Store this episode's observed return.
        value_history[episode] = estimated_values  # Store the critic's complete value snapshot.
        observer(
            TrainingSnapshot(
                episode,
                trajectory[-1],
                tuple(trajectory),
                state_values=estimated_values,
                action_probabilities=probabilities,
            )
        )

    return TrainingResult(probabilities, episode_returns, value_history)


def main() -> None:
    import tyro

    from rltoy.cli.common import run

    @dataclass(frozen=True)
    class GAERunConfig(GAEConfig):
        environment_id: str = "RLToy/BranchingRisk-v0"
        graph_path: Path | None = None
        seed: int | None = 0
        render: bool = False
        render_every_steps: int = 1
        render_delay_ms: int = 50

    run(train, tyro.cli(GAERunConfig), "GAE actor-critic")


if __name__ == "__main__":
    main()
