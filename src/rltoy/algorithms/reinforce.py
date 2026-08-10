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
    returns = [0.0] * len(rewards)
    future_return = 0.0
    for step in range(len(rewards) - 1, -1, -1):
        future_return = rewards[step] + gamma * future_return
        returns[step] = future_return
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
        torch.manual_seed(seed)
    device = torch.device(config.device)
    states, actions = env.observation_space.n, env.action_space.n
    policy = PolicyNetwork(states, actions, config.hidden_units).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    returns = np.zeros(config.episodes)
    state_values = np.zeros((config.episodes, states))

    for episode in range(config.episodes):
        state, _ = env.reset(seed=seed if episode == 0 else None)
        trajectory = [state]
        rewards = []
        log_probabilities = []

        while True:
            logits = policy(one_hot(np.asarray([state]), states, device))[0]
            distribution = torch.distributions.Categorical(logits=logits)
            action = distribution.sample()
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            log_probabilities.append(distribution.log_prob(action))
            rewards.append(reward)
            trajectory.append(next_state)
            if done:
                break
            state = next_state

        # Vanilla REINFORCE weighs each sampled action by its full return-to-go.
        episode_returns = torch.as_tensor(discounted_returns(rewards, config.gamma), device=device)
        loss = -(torch.stack(log_probabilities) * episode_returns).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            probabilities = torch.softmax(policy(one_hot(np.arange(states), states, device)), dim=1).cpu().numpy()
        returns[episode] = sum(rewards)
        state_values[episode] = probabilities.max(axis=1)
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
