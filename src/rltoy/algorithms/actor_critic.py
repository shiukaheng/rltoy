"""One-step TD actor-critic, written to show its online update rule directly."""

from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from rltoy.algorithms.tabular import Observer, TrainingResult, TrainingSnapshot, observe_nothing


@dataclass(frozen=True)
class ActorCriticConfig:
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


def train(
    env: gym.Env,
    config: ActorCriticConfig,
    seed: int | None = None,
    observer: Observer = observe_nothing,
) -> TrainingResult:
    """Learn a policy from one-step TD errors, updating after every transition."""
    if not isinstance(env.observation_space, gym.spaces.Discrete):
        raise ValueError("Actor-critic currently requires a discrete observation space")
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise ValueError("Actor-critic currently requires a discrete action space")

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
        episode_return = 0.0

        while True:
            state_tensor = one_hot(np.asarray([state]), states, device)
            logits = policy(state_tensor)[0]
            distribution = torch.distributions.Categorical(logits=logits)
            action = distribution.sample()
            state_value = value(state_tensor)[0]
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            with torch.no_grad():
                next_value = torch.zeros((), device=device)
                if not done:
                    next_value = value(one_hot(np.asarray([next_state]), states, device))[0]
                td_target = torch.as_tensor(reward, dtype=torch.float32, device=device) + config.gamma * next_value
            td_error = td_target - state_value

            actor_loss = -distribution.log_prob(action) * td_error.detach()
            critic_loss = 0.5 * td_error.square()
            policy_optimizer.zero_grad()
            actor_loss.backward()
            policy_optimizer.step()
            value_optimizer.zero_grad()
            critic_loss.backward()
            value_optimizer.step()

            episode_return += reward
            trajectory.append(next_state)
            with torch.no_grad():
                all_states = one_hot(np.arange(states), states, device)
                probabilities = torch.softmax(policy(all_states), dim=1).cpu().numpy()
                estimated_values = value(all_states).cpu().numpy()
            observer(
                TrainingSnapshot(
                    episode,
                    next_state,
                    tuple(trajectory),
                    state_values=estimated_values,
                    action_probabilities=probabilities,
                )
            )
            if done:
                break
            state = next_state

        episode_returns[episode] = episode_return
        value_history[episode] = estimated_values

    return TrainingResult(probabilities, episode_returns, value_history)


def main() -> None:
    import tyro

    from rltoy.cli.common import run

    @dataclass(frozen=True)
    class ActorCriticRunConfig(ActorCriticConfig):
        environment_id: str = "RLToy/BranchingRisk-v0"
        graph_path: Path | None = None
        seed: int | None = 0
        render: bool = False
        render_every_steps: int = 1
        render_delay_ms: int = 50

    run(train, tyro.cli(ActorCriticRunConfig), "One-step TD actor-critic")


if __name__ == "__main__":
    main()
