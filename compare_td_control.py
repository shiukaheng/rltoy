from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro

from bettermdptools.algorithms.planner import Planner
from frozen_lake import make_frozen_lake
from graph_world import make_graph_world


@dataclass
class Config:
    env: str = "frozen_lake"
    graph_path: Path = Path("graphs/risk_vs_delay.json")
    n_episodes: int = 500
    n_runs: int = 20
    gamma: float = 0.99
    trace_decay: float = 0.8
    alpha_init: float = 0.5
    alpha_min: float = 0.01
    alpha_decay_ratio: float = 0.5
    epsilon_init: float = 1.0
    epsilon_min: float = 0.1
    epsilon_decay_ratio: float = 0.9
    step_cost: float | None = None
    is_slippery: bool = False
    seed: int = 0
    output_path: Path = Path("td_control_comparison.png")
    show: bool = False


def decay_schedule(initial: float, minimum: float, ratio: float, steps: int) -> np.ndarray:
    if steps < 1:
        raise ValueError("n_episodes must be at least 1")
    if steps == 1:
        return np.array([initial])

    decay_steps = min(steps, max(2, int(steps * ratio)))
    values = np.logspace(-2, 0, decay_steps)[::-1]
    values = (values - values.min()) / (values.max() - values.min())
    values = (initial - minimum) * values + minimum
    return np.pad(values, (0, steps - decay_steps), mode="edge")


def train(
    algorithm: str,
    env,
    ground_truth_v: np.ndarray,
    valid_states: np.ndarray,
    config: Config,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions), dtype=np.float64)
    losses = np.empty(config.n_episodes)
    alphas = decay_schedule(config.alpha_init, config.alpha_min, config.alpha_decay_ratio, config.n_episodes)
    epsilons = decay_schedule(config.epsilon_init, config.epsilon_min, config.epsilon_decay_ratio, config.n_episodes)

    def select_action(state: int, epsilon: float) -> int:
        if rng.random() < epsilon:
            return int(rng.integers(n_actions))
        values = Q[state]
        return int(rng.choice(np.flatnonzero(values == values.max())))

    for episode in range(config.n_episodes):
        traces = np.zeros_like(Q) if algorithm == "SARSA(lambda)" else None
        state, _ = env.reset(seed=seed if episode == 0 else None)
        terminated, truncated = False, False
        action = select_action(state, epsilons[episode])

        while not (terminated or truncated):
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if algorithm == "Q-learning":
                td_target = reward + config.gamma * np.max(Q[next_state]) * (not done)
                Q[state, action] += alphas[episode] * (td_target - Q[state, action])
                state = next_state
                action = select_action(state, epsilons[episode])
            else:
                next_action = select_action(next_state, epsilons[episode])
                td_error = (
                    reward
                    + config.gamma * Q[next_state, next_action] * (not done)
                    - Q[state, action]
                )
                if algorithm == "SARSA(lambda)":
                    traces[state, action] += 1
                    Q += alphas[episode] * td_error * traces
                    traces *= config.gamma * config.trace_decay
                else:
                    Q[state, action] += alphas[episode] * td_error
                state, action = next_state, next_action

        learned_v = np.max(Q, axis=1)
        losses[episode] = np.mean((learned_v[valid_states] - ground_truth_v[valid_states]) ** 2)

    return losses


def _make_env(config: Config) -> tuple:
    if config.env == "frozen_lake":
        env, _ = make_frozen_lake(is_slippery=config.is_slippery)
        ground_truth_v, _, _ = Planner(env.unwrapped.P).value_iteration(gamma=config.gamma)
        ground_truth_v = np.asarray(ground_truth_v, dtype=np.float64)
        valid_states = (env.unwrapped.desc.reshape(-1) != b"H")
    else:
        env, _ = make_graph_world(config.graph_path, gamma=config.gamma, step_cost=config.step_cost)
        ground_truth_v, _, _ = Planner(env.P).value_iteration(gamma=config.gamma)
        ground_truth_v = np.asarray(ground_truth_v, dtype=np.float64)
        valid_states = ~env.terminal_states()
    return env, ground_truth_v, valid_states


def main(config: Config) -> None:
    _, ground_truth_v, valid_states = _make_env(config)

    algorithms = ("SARSA", "SARSA(lambda)", "Q-learning")
    losses_by_algorithm = {
        algorithm: np.empty((config.n_runs, config.n_episodes))
        for algorithm in algorithms
    }

    for run in range(config.n_runs):
        for algorithm in algorithms:
            env, _, _ = _make_env(config)
            losses_by_algorithm[algorithm][run] = train(
                algorithm, env, ground_truth_v, valid_states, config, config.seed + run,
            )
            env.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    episodes = np.arange(1, config.n_episodes + 1)
    for algorithm, losses in losses_by_algorithm.items():
        mean = losses.mean(axis=0)
        stderr = losses.std(axis=0) / np.sqrt(config.n_runs)
        line = ax.plot(episodes, mean, label=algorithm)[0]
        ax.fill_between(episodes, mean - stderr, mean + stderr, alpha=0.2, color=line.get_color())

    env_label = "FrozenLake" if config.env == "frozen_lake" else "GraphWorld"
    ax.set(
        title=f"TD control convergence on {env_label}",
        xlabel="Training episode",
        ylabel="MSE of learned V against planner V",
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    fig.savefig(config.output_path, dpi=150)
    print(f"Saved {config.output_path}")
    if config.show:
        plt.show()


if __name__ == "__main__":
    main(tyro.cli(Config))