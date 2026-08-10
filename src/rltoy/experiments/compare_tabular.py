"""Compare the real tabular learners against a graph planner oracle."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro

from rltoy.algorithms import q_learning, sarsa, sarsa_lambda
from rltoy.algorithms.sarsa_lambda import SarsaLambdaConfig
from rltoy.algorithms.tabular import TabularConfig
from rltoy.envs.graph_world import GraphWorldEnv, planner_values


@dataclass(frozen=True)
class Config(TabularConfig):
    graph_path: Path | None = None
    runs: int = 20
    seed: int = 0
    trace_decay: float = 0.8
    output_path: Path = Path("td_control_comparison.png")
    show: bool = False


def make_environment(graph_path: Path | None) -> GraphWorldEnv:
    if graph_path is not None:
        return GraphWorldEnv.from_json(graph_path)
    from rltoy.envs.graph_world import make_branching_risk

    return make_branching_risk()


def main() -> None:
    config = tyro.cli(Config)
    reference_env = make_environment(config.graph_path)
    optimal_values = planner_values(reference_env, config.gamma)
    nonterminal = ~reference_env.terminal_states()
    learners = {
        "Q-learning": (q_learning.train, config),
        "SARSA": (sarsa.train, config),
        "SARSA(lambda)": (
            sarsa_lambda.train,
            SarsaLambdaConfig(
                episodes=config.episodes,
                gamma=config.gamma,
                alpha_initial=config.alpha_initial,
                alpha_final=config.alpha_final,
                alpha_decay_ratio=config.alpha_decay_ratio,
                epsilon_initial=config.epsilon_initial,
                epsilon_final=config.epsilon_final,
                epsilon_decay_ratio=config.epsilon_decay_ratio,
                trace_decay=config.trace_decay,
            ),
        ),
    }
    losses = {name: np.empty((config.runs, config.episodes)) for name in learners}

    for run_index in range(config.runs):
        for name, (train, learner_config) in learners.items():
            env = make_environment(config.graph_path)
            try:
                result = train(env, learner_config, config.seed + run_index)
            finally:
                env.close()
            losses[name][run_index] = np.mean(
                (result.state_values[:, nonterminal] - optimal_values[nonterminal]) ** 2,
                axis=1,
            )

    figure, axis = plt.subplots(figsize=(10, 6))
    episodes = np.arange(1, config.episodes + 1)
    for name, values in losses.items():
        mean = values.mean(axis=0)
        stderr = values.std(axis=0) / np.sqrt(config.runs)
        line = axis.plot(episodes, mean, label=name)[0]
        axis.fill_between(episodes, mean - stderr, mean + stderr, color=line.get_color(), alpha=0.2)
    axis.set(title="Tabular TD control on Branching Risk", xlabel="Episode", ylabel="MSE against optimal V")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(config.output_path, dpi=150)
    print(f"Saved {config.output_path}")
    if config.show:
        plt.show()
    reference_env.close()
