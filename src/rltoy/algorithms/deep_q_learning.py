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
        self.transitions.append((state, action, reward, next_state, done))  # Retain this experience, discarding the oldest when full.

    def sample(self, batch_size: int, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
        indices = rng.choice(len(self.transitions), size=batch_size, replace=False)  # Choose distinct past experiences uniformly.
        batch = [self.transitions[index] for index in indices]  # Gather the selected transitions.
        return tuple(np.asarray(values) for values in zip(*batch, strict=True))  # Group the batch into aligned arrays by field.

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
    state_ids = torch.as_tensor(states, dtype=torch.long, device=device)  # Convert discrete state IDs to tensor indices.
    return torch.nn.functional.one_hot(state_ids, state_count).to(torch.float32)  # Encode each state as the network input vector.


def epsilon_greedy(q_values: torch.Tensor, epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(q_values.numel()))  # Explore by selecting a uniformly random action.
    return int(torch.argmax(q_values).item())  # Exploit the online network's highest-valued action.


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

    rng = np.random.default_rng(seed)  # Create reproducible random choices for replay and exploration.
    if seed is not None:
        torch.manual_seed(seed)  # Make network initialization and sampled actions reproducible.
    device = torch.device(config.device)  # Place model computations on the requested device.
    states, actions = env.observation_space.n, env.action_space.n  # Read the one-hot input and action-output dimensions.
    online = QNetwork(states, actions, config.hidden_units).to(device)  # Create the network being optimized.
    target = QNetwork(states, actions, config.hidden_units).to(device)  # Create a stable network for bootstrap targets.
    target.load_state_dict(online.state_dict())  # Start both networks with identical action values.
    optimizer = torch.optim.Adam(online.parameters(), lr=config.learning_rate)  # Optimize only the online network.
    replay = ReplayBuffer(config.replay_capacity)  # Store past transitions for decorrelated updates.
    returns = np.zeros(config.episodes)  # Reserve one undiscounted return for each episode.
    state_values = np.zeros((config.episodes, states))  # Record V(s) = max_a Q(s, a) after each episode.
    epsilons = decay_schedule(config.epsilon_initial, config.epsilon_final, config.epsilon_decay_ratio, config.episodes)  # Schedule exploration.
    updates = 0  # Count gradient updates to know when to refresh the target network.

    for episode, epsilon in enumerate(epsilons):  # Collect experience using each scheduled exploration rate.
        state, _ = env.reset(seed=seed if episode == 0 else None)  # Begin a new episode, seeding only the first reset.
        trajectory = [state]  # Track visited states for the optional observer.
        episode_return = 0.0  # Accumulate this episode's actual rewards.

        while True:
            with torch.no_grad():  # Action selection does not need an autograd graph.
                q_values = online(one_hot(np.asarray([state]), states, device))[0]  # Estimate Q(s, .) with the online network.
            action = epsilon_greedy(q_values, epsilon, rng)  # Choose an exploratory or greedy behavior action.
            next_state, reward, terminated, truncated, _ = env.step(action)  # Sample the environment transition.
            done = terminated or truncated  # Treat natural endings and time limits as episode boundaries.
            replay.add(state, action, reward, next_state, done)  # Save this experience for future minibatches.

            if len(replay) >= config.batch_size:
                batch_states, batch_actions, rewards, next_states, dones = replay.sample(config.batch_size, rng)  # Draw an independent replay minibatch.
                current_q = online(one_hot(batch_states, states, device)).gather(
                    1, torch.as_tensor(batch_actions, device=device).unsqueeze(1)
                ).squeeze(1)  # Select Q(s, a) for each action actually taken.
                with torch.no_grad():  # Targets stay fixed while differentiating the online network.
                    next_q = target(one_hot(next_states, states, device)).max(dim=1).values  # Estimate max_a' Q_target(s', a').
                    targets = torch.as_tensor(rewards, dtype=torch.float32, device=device) + config.gamma * next_q * (
                        1 - torch.as_tensor(dones, dtype=torch.float32, device=device)
                    )  # Add the discounted target value only for nonterminal transitions.
                loss = torch.nn.functional.mse_loss(current_q, targets)  # Penalize disagreement with Bellman targets.
                optimizer.zero_grad()  # Clear gradients from the preceding minibatch.
                loss.backward()  # Differentiate the Q-learning loss through the online network.
                optimizer.step()  # Improve the online action-value approximator.
                updates += 1  # Advance the target-network refresh counter.
                if updates % config.target_update_interval == 0:
                    target.load_state_dict(online.state_dict())  # Periodically make bootstrap targets follow the online network.

            episode_return += reward  # Add the transition reward to the episode total.
            trajectory.append(next_state)  # Extend the recorded state path.
            with torch.no_grad():  # Reporting values does not need an autograd graph.
                all_q_values = online(one_hot(np.arange(states), states, device)).cpu().numpy()  # Estimate every state's action values for reporting.
            observer(
                TrainingSnapshot(
                    episode, next_state, tuple(trajectory), action_values=all_q_values
                )
            )
            if done:
                break
            state = next_state  # Continue from the sampled successor state.

        returns[episode] = episode_return  # Store this episode's observed return.
        state_values[episode] = all_q_values.max(axis=1)  # Derive a greedy state value from each network output row.

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
