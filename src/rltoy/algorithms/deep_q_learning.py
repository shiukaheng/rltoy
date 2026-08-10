"""Deep Q-learning (DQN), written to follow its textbook pseudocode."""

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from rltoy.algorithms.tabular import (
    Observer,
    TabularConfig,
    TrainingResult,
    TrainingSnapshot,
    decay_schedule,
    observe_nothing,
)


@dataclass(frozen=True)
class DeepQConfig(TabularConfig):
    hidden_units: int = 32
    replay_capacity: int = 10_000
    batch_size: int = 32
    learning_rate: float = 1e-3
    target_update_interval: int = 100
    device: str = "cpu"


class ReplayBuffer:
    """Fixed-size experience store used by DQN's replay updates."""

    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("replay_capacity must be at least 1")
        self.transitions: deque[tuple[int, int, float, int, bool]] = deque(maxlen=capacity)

    def add(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        self.transitions.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
        indices = rng.choice(len(self.transitions), size=batch_size, replace=False)
        batch = [self.transitions[index] for index in indices]
        return tuple(np.asarray(values) for values in zip(*batch, strict=True))

    def __len__(self) -> int:
        return len(self.transitions)


class QNetwork(nn.Module):
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


def epsilon_greedy(q_values: torch.Tensor, epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(q_values.numel()))
    return int(torch.argmax(q_values).item())


def train(
    env: gym.Env,
    config: DeepQConfig,
    seed: int | None = None,
    observer: Observer = observe_nothing,
) -> TrainingResult:
    """Learn action values with DQN replay and a periodically copied target network."""
    if config.batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if config.target_update_interval < 1:
        raise ValueError("target_update_interval must be at least 1")
    if not isinstance(env.observation_space, gym.spaces.Discrete):
        raise ValueError("Deep Q-learning currently requires a discrete observation space")
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise ValueError("Deep Q-learning currently requires a discrete action space")

    rng = np.random.default_rng(seed)
    if seed is not None:
        torch.manual_seed(seed)
    device = torch.device(config.device)
    states, actions = env.observation_space.n, env.action_space.n
    online = QNetwork(states, actions, config.hidden_units).to(device)
    target = QNetwork(states, actions, config.hidden_units).to(device)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=config.learning_rate)
    replay = ReplayBuffer(config.replay_capacity)
    returns = np.zeros(config.episodes)
    state_values = np.zeros((config.episodes, states))
    epsilons = decay_schedule(config.epsilon_initial, config.epsilon_final, config.epsilon_decay_ratio, config.episodes)
    updates = 0

    for episode, epsilon in enumerate(epsilons):
        state, _ = env.reset(seed=seed if episode == 0 else None)
        trajectory = [state]
        episode_return = 0.0

        while True:
            with torch.no_grad():
                q_values = online(one_hot(np.asarray([state]), states, device))[0]
            action = epsilon_greedy(q_values, epsilon, rng)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay.add(state, action, reward, next_state, done)

            if len(replay) >= config.batch_size:
                batch_states, batch_actions, rewards, next_states, dones = replay.sample(config.batch_size, rng)
                current_q = online(one_hot(batch_states, states, device)).gather(
                    1, torch.as_tensor(batch_actions, device=device).unsqueeze(1)
                ).squeeze(1)
                with torch.no_grad():
                    next_q = target(one_hot(next_states, states, device)).max(dim=1).values
                    targets = torch.as_tensor(rewards, dtype=torch.float32, device=device) + config.gamma * next_q * (
                        1 - torch.as_tensor(dones, dtype=torch.float32, device=device)
                    )
                loss = torch.nn.functional.mse_loss(current_q, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                updates += 1
                if updates % config.target_update_interval == 0:
                    target.load_state_dict(online.state_dict())

            episode_return += reward
            trajectory.append(next_state)
            with torch.no_grad():
                all_q_values = online(one_hot(np.arange(states), states, device)).cpu().numpy()
            observer(TrainingSnapshot(episode, next_state, all_q_values, tuple(trajectory)))
            if done:
                break
            state = next_state

        returns[episode] = episode_return
        state_values[episode] = all_q_values.max(axis=1)

    return TrainingResult(all_q_values, returns, state_values)


def main() -> None:
    import tyro

    from rltoy.cli.common import run

    @dataclass(frozen=True)
    class DeepQRunConfig(DeepQConfig):
        environment_id: str = "RLToy/BranchingRisk-v0"
        graph_path: Path | None = None
        seed: int | None = 0
        render: bool = False
        render_every_steps: int = 1
        render_delay_ms: int = 50

    run(train, tyro.cli(DeepQRunConfig), "Deep Q-learning")


if __name__ == "__main__":
    main()
