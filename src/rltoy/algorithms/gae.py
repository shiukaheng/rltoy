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
    advantages = np.zeros(len(rewards))
    future_advantage = 0.0
    for step in range(len(rewards) - 1, -1, -1):
        is_final_step = step == len(rewards) - 1
        next_value = bootstrap_value if is_final_step else values[step + 1]
        bootstrap = 0.0 if is_final_step and terminated else 1.0
        continuation = 0.0 if is_final_step else 1.0
        td_error = rewards[step] + gamma * bootstrap * next_value - values[step]
        future_advantage = td_error + gamma * gae_lambda * continuation * future_advantage
        advantages[step] = future_advantage
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
        torch.manual_seed(seed)
    device = torch.device(config.device)
    states, actions = env.observation_space.n, env.action_space.n
    policy = PolicyNetwork(states, actions, config.hidden_units).to(device)
    value = ValueNetwork(states, config.hidden_units).to(device)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=config.policy_learning_rate)
    value_optimizer = torch.optim.Adam(value.parameters(), lr=config.value_learning_rate)
    episode_returns = np.zeros(config.episodes)
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
            visited_states.append(state)
            rewards.append(reward)
            log_probabilities.append(distribution.log_prob(action))
            trajectory.append(next_state)
            if terminated or truncated:
                break
            state = next_state

        with torch.no_grad():
            visited = one_hot(np.asarray(visited_states), states, device)
            old_values = value(visited).cpu().numpy()
            bootstrap_value = 0.0 if terminated else value(
                one_hot(np.asarray([next_state]), states, device)
            ).item()
        advantages = generalized_advantages(
            rewards, old_values, bootstrap_value, config.gamma, config.gae_lambda, terminated
        )
        returns_to_go = advantages + old_values
        advantages_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        returns_tensor = torch.as_tensor(returns_to_go, dtype=torch.float32, device=device)
        policy_loss = -(torch.stack(log_probabilities) * advantages_tensor).sum()
        value_loss = torch.nn.functional.mse_loss(value(visited), returns_tensor)

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
        episode_returns[episode] = sum(rewards)
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
