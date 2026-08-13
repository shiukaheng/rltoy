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
        torch.manual_seed(seed)  # Make network initialization and action samples reproducible.
    device = torch.device(config.device)  # Place network computations on the requested device.
    states, actions = env.observation_space.n, env.action_space.n  # Read the one-hot input and action-output dimensions.
    policy = PolicyNetwork(states, actions, config.hidden_units).to(device)  # Create the stochastic actor.
    value = ValueNetwork(states, config.hidden_units).to(device)  # Create the state-value critic.
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=config.policy_learning_rate)  # Optimize the actor separately.
    value_optimizer = torch.optim.Adam(value.parameters(), lr=config.value_learning_rate)  # Optimize the critic separately.
    episode_returns = np.zeros(config.episodes)  # Reserve one observed return for each episode.
    value_history = np.zeros((config.episodes, states))  # Record the critic's estimate for every state.

    for episode in range(config.episodes):  # Update both networks after every transition in an episode.
        state, _ = env.reset(seed=seed if episode == 0 else None)  # Begin a new episode, seeding only the first reset.
        trajectory = [state]  # Track visited states for the optional observer.
        episode_return = 0.0  # Accumulate this episode's actual rewards.

        while True:
            state_tensor = one_hot(np.asarray([state]), states, device)  # Encode the current discrete state for both networks.
            logits = policy(state_tensor)[0]  # Produce unnormalized action preferences.
            distribution = torch.distributions.Categorical(logits=logits)  # Turn preferences into a stochastic policy.
            action = distribution.sample()  # Sample the behavior action from the actor.
            state_value = value(state_tensor)[0]  # Estimate V(s) with the critic.
            next_state, reward, terminated, truncated, _ = env.step(action.item())  # Sample the environment transition.
            done = terminated or truncated  # Treat natural endings and time limits as episode boundaries.

            with torch.no_grad():  # Bootstrap targets must not backpropagate through the next state.
                next_value = torch.zeros((), device=device)  # Terminal transitions have zero successor value.
                if not done:
                    next_value = value(one_hot(np.asarray([next_state]), states, device))[0]  # Bootstrap from the critic's next-state estimate.
                td_target = torch.as_tensor(reward, dtype=torch.float32, device=device) + config.gamma * next_value  # Form the one-step TD target.
            td_error = td_target - state_value  # Measure the critic's one-step prediction error.

            actor_loss = -distribution.log_prob(action) * td_error.detach()  # Reinforce this action according to a fixed TD advantage.
            critic_loss = 0.5 * td_error.square()  # Penalize the critic's TD prediction error.
            policy_optimizer.zero_grad()  # Clear actor gradients from the prior transition.
            actor_loss.backward()  # Differentiate the actor loss.
            policy_optimizer.step()  # Update the policy immediately.
            value_optimizer.zero_grad()  # Clear critic gradients from the prior transition.
            critic_loss.backward()  # Differentiate the critic loss.
            value_optimizer.step()  # Update the state-value estimate immediately.

            episode_return += reward  # Add the transition reward to the episode total.
            trajectory.append(next_state)  # Extend the recorded state path.
            with torch.no_grad():  # Reporting values and probabilities does not need an autograd graph.
                all_states = one_hot(np.arange(states), states, device)  # Encode every state for a complete snapshot.
                probabilities = torch.softmax(policy(all_states), dim=1).cpu().numpy()  # Evaluate pi(a | s) for every state.
                estimated_values = value(all_states).cpu().numpy()  # Evaluate V(s) for every state.
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
            state = next_state  # Continue from the sampled successor state.

        episode_returns[episode] = episode_return  # Store this episode's observed return.
        value_history[episode] = estimated_values  # Store the critic's complete value snapshot.

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
