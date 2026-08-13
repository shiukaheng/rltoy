"""Vanilla REINFORCE, written to follow its textbook pseudocode."""

from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from rltoy.algorithms.tabular import Observer, TrainingResult, TrainingSnapshot, observe_nothing


@dataclass(frozen=True)
class ReinforceConfig:
    episodes: int = 1_000
    gamma: float = 0.99
    hidden_units: int = 32
    learning_rate: float = 1e-3
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
    config: ReinforceConfig,
    seed: int | None = None,
    observer: Observer = observe_nothing,
) -> TrainingResult:
    """Learn a stochastic policy from complete episodes, without a baseline."""
    if not isinstance(env.observation_space, gym.spaces.Discrete):
        raise ValueError("REINFORCE currently requires a discrete observation space")
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise ValueError("REINFORCE currently requires a discrete action space")

    if seed is not None:
        torch.manual_seed(seed)  # Make policy initialization and action samples reproducible.
    device = torch.device(config.device)  # Place policy computations on the requested device.
    states, actions = env.observation_space.n, env.action_space.n  # Read the one-hot input and action-output dimensions.
    policy = PolicyNetwork(states, actions, config.hidden_units).to(device)  # Create the stochastic policy being optimized.
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)  # Optimize the policy parameters.
    returns = np.zeros(config.episodes)  # Reserve one observed return for each episode.
    state_values = np.zeros((config.episodes, states))  # Store each state's largest action probability for reporting.

    for episode in range(config.episodes):  # Collect and learn from one complete policy episode.
        state, _ = env.reset(seed=seed if episode == 0 else None)  # Begin a new episode, seeding only the first reset.
        trajectory = [state]  # Track visited states for the optional observer.
        rewards = []  # Keep rewards until their return-to-go can be computed.
        log_probabilities = []  # Keep differentiable log probabilities of sampled actions.

        while True:
            logits = policy(one_hot(np.asarray([state]), states, device))[0]  # Produce unnormalized action preferences for this state.
            distribution = torch.distributions.Categorical(logits=logits)  # Turn preferences into a categorical policy.
            action = distribution.sample()  # Sample an action from the current policy.
            next_state, reward, terminated, truncated, _ = env.step(action.item())  # Sample the environment transition.
            done = terminated or truncated  # Treat natural endings and time limits as episode boundaries.
            log_probabilities.append(distribution.log_prob(action))  # Retain the score-function term for this action.
            rewards.append(reward)  # Retain the reward for the later return calculation.
            trajectory.append(next_state)  # Extend the recorded state path.
            if done:
                break
            state = next_state  # Continue from the sampled successor state.

        # Vanilla REINFORCE weighs each sampled action by its full return-to-go.
        episode_returns = torch.as_tensor(discounted_returns(rewards, config.gamma), device=device)  # Compute every sampled action's discounted return-to-go.
        loss = -(torch.stack(log_probabilities) * episode_returns).sum()  # Increase log probability for actions with high returns.
        optimizer.zero_grad()  # Clear gradients from the previous episode.
        loss.backward()  # Differentiate the policy-gradient estimator.
        optimizer.step()  # Update the policy after the complete episode.

        with torch.no_grad():  # Reporting policy probabilities does not need an autograd graph.
            probabilities = torch.softmax(policy(one_hot(np.arange(states), states, device)), dim=1).cpu().numpy()  # Evaluate pi(a | s) for every state.
        returns[episode] = sum(rewards)  # Store this episode's observed return.
        state_values[episode] = probabilities.max(axis=1)  # Report the probability of the most likely action per state.
        observer(
            TrainingSnapshot(
                episode,
                trajectory[-1],
                tuple(trajectory),
                action_probabilities=probabilities,
            )
        )

    return TrainingResult(probabilities, returns, state_values)


def main() -> None:
    import tyro

    from rltoy.cli.common import run

    @dataclass(frozen=True)
    class ReinforceRunConfig(ReinforceConfig):
        environment_id: str = "RLToy/BranchingRisk-v0"
        graph_path: Path | None = None
        seed: int | None = 0
        render: bool = False
        render_every_steps: int = 1
        render_delay_ms: int = 50

    run(train, tyro.cli(ReinforceRunConfig), "Vanilla REINFORCE")


if __name__ == "__main__":
    main()
