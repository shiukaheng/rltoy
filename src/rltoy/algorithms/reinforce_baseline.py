"""REINFORCE with a learned state-value baseline, kept separate from vanilla REINFORCE."""

from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from rltoy.algorithms.tabular import Observer, TrainingResult, TrainingSnapshot, observe_nothing


@dataclass(frozen=True)
class ReinforceBaselineConfig:
    episodes: int = 1_000
    gamma: float = 0.99
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


def discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    returns = [0.0] * len(rewards)  # Allocate one return-to-go for every sampled action.
    future_return = 0.0  # The return after the terminal transition is zero.
    for step in range(len(rewards) - 1, -1, -1):  # Work backward so each suffix return is already known.
        future_return = rewards[step] + gamma * future_return  # Add this reward to the discounted future rewards.
        returns[step] = future_return  # Save the return-to-go from this time step.
    return returns


def train(
    env: gym.Env,
    config: ReinforceBaselineConfig,
    seed: int | None = None,
    observer: Observer = observe_nothing,
) -> TrainingResult:
    """Learn a policy using returns-to-go minus a learned V(s) baseline."""
    if not isinstance(env.observation_space, gym.spaces.Discrete):
        raise ValueError("REINFORCE with baseline currently requires a discrete observation space")
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise ValueError("REINFORCE with baseline currently requires a discrete action space")

    if seed is not None:
        torch.manual_seed(seed)  # Make network initialization and action samples reproducible.
    device = torch.device(config.device)  # Place network computations on the requested device.
    states, actions = env.observation_space.n, env.action_space.n  # Read the one-hot input and action-output dimensions.
    policy = PolicyNetwork(states, actions, config.hidden_units).to(device)  # Create the stochastic actor.
    value = ValueNetwork(states, config.hidden_units).to(device)  # Create the state-value baseline.
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=config.policy_learning_rate)  # Optimize the actor separately.
    value_optimizer = torch.optim.Adam(value.parameters(), lr=config.value_learning_rate)  # Optimize the baseline separately.
    episode_rewards = np.zeros(config.episodes)  # Reserve one observed return for each episode.
    value_history = np.zeros((config.episodes, states))  # Record the critic's estimate for every state.

    for episode in range(config.episodes):  # Collect and learn from one complete policy episode.
        state, _ = env.reset(seed=seed if episode == 0 else None)  # Begin a new episode, seeding only the first reset.
        trajectory = [state]  # Track visited states for the optional observer.
        visited_states = []  # Keep states aligned with sampled actions and returns.
        rewards = []  # Keep rewards until their return-to-go can be computed.
        log_probabilities = []  # Keep differentiable log probabilities of sampled actions.

        while True:
            logits = policy(one_hot(np.asarray([state]), states, device))[0]  # Produce unnormalized action preferences for this state.
            distribution = torch.distributions.Categorical(logits=logits)  # Turn preferences into a categorical policy.
            action = distribution.sample()  # Sample an action from the current policy.
            next_state, reward, terminated, truncated, _ = env.step(action.item())  # Sample the environment transition.
            done = terminated or truncated  # Treat natural endings and time limits as episode boundaries.
            visited_states.append(state)  # Retain the state whose action received this reward sequence.
            log_probabilities.append(distribution.log_prob(action))  # Retain the score-function term for this action.
            rewards.append(reward)  # Retain the reward for the later return calculation.
            trajectory.append(next_state)  # Extend the recorded state path.
            if done:
                break
            state = next_state  # Continue from the sampled successor state.

        returns_to_go = torch.as_tensor(discounted_returns(rewards, config.gamma), device=device)  # Compute every sampled action's discounted return-to-go.
        baseline = value(one_hot(np.asarray(visited_states), states, device))  # Estimate V(s) for every visited state.
        advantages = returns_to_go - baseline.detach()  # Measure returns above or below the baseline without training it through the actor.
        policy_loss = -(torch.stack(log_probabilities) * advantages).sum()  # Increase action likelihood in proportion to estimated advantage.
        value_loss = torch.nn.functional.mse_loss(baseline, returns_to_go)  # Train the baseline to predict sampled returns.

        policy_optimizer.zero_grad()  # Clear actor gradients from the previous episode.
        policy_loss.backward()  # Differentiate the baseline-adjusted policy-gradient estimator.
        policy_optimizer.step()  # Update the policy.
        value_optimizer.zero_grad()  # Clear critic gradients from the previous episode.
        value_loss.backward()  # Differentiate the value prediction error.
        value_optimizer.step()  # Update the state-value baseline.

        with torch.no_grad():  # Reporting values and probabilities does not need an autograd graph.
            all_states = one_hot(np.arange(states), states, device)  # Encode every state for a complete policy and value snapshot.
            probabilities = torch.softmax(policy(all_states), dim=1).cpu().numpy()  # Evaluate pi(a | s) for every state.
            estimated_values = value(all_states).cpu().numpy()  # Evaluate V(s) for every state.
        episode_rewards[episode] = sum(rewards)  # Store this episode's observed return.
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

    return TrainingResult(probabilities, episode_rewards, value_history)


def main() -> None:
    import tyro

    from rltoy.cli.common import run

    @dataclass(frozen=True)
    class ReinforceBaselineRunConfig(ReinforceBaselineConfig):
        environment_id: str = "RLToy/BranchingRisk-v0"
        graph_path: Path | None = None
        seed: int | None = 0
        render: bool = False
        render_every_steps: int = 1
        render_delay_ms: int = 50

    run(train, tyro.cli(ReinforceBaselineRunConfig), "REINFORCE with baseline")


if __name__ == "__main__":
    main()
