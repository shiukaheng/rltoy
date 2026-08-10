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
    returns = [0.0] * len(rewards)
    future_return = 0.0
    for step in range(len(rewards) - 1, -1, -1):
        future_return = rewards[step] + gamma * future_return
        returns[step] = future_return
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
        torch.manual_seed(seed)
    device = torch.device(config.device)
    states, actions = env.observation_space.n, env.action_space.n
    policy = PolicyNetwork(states, actions, config.hidden_units).to(device)
    value = ValueNetwork(states, config.hidden_units).to(device)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=config.policy_learning_rate)
    value_optimizer = torch.optim.Adam(value.parameters(), lr=config.value_learning_rate)
    episode_rewards = np.zeros(config.episodes)
    value_history = np.zeros((config.episodes, states))

    for episode in range(config.episodes):
        state, _ = env.reset(seed=seed if episode == 0 else None)
        trajectory = [state]
        visited_states = []
        rewards = []
        log_probabilities = []

        while True:
            logits = policy(one_hot(np.asarray([state]), states, device))[0]
            distribution = torch.distributions.Categorical(logits=logits)
            action = distribution.sample()
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            visited_states.append(state)
            log_probabilities.append(distribution.log_prob(action))
            rewards.append(reward)
            trajectory.append(next_state)
            if done:
                break
            state = next_state

        returns_to_go = torch.as_tensor(discounted_returns(rewards, config.gamma), device=device)
        baseline = value(one_hot(np.asarray(visited_states), states, device))
        advantages = returns_to_go - baseline.detach()
        policy_loss = -(torch.stack(log_probabilities) * advantages).sum()
        value_loss = torch.nn.functional.mse_loss(baseline, returns_to_go)

        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        with torch.no_grad():
            all_states = one_hot(np.arange(states), states, device)
            probabilities = torch.softmax(policy(all_states), dim=1).cpu().numpy()
            estimated_values = value(all_states).cpu().numpy()
        episode_rewards[episode] = sum(rewards)
        value_history[episode] = estimated_values
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
