"""CLI wiring kept outside the readable algorithm modules."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import gymnasium as gym

from rltoy.algorithms.tabular import TabularConfig, TrainingResult
from rltoy.envs.graph_world import GraphWorldEnv
from rltoy.visualization.graph_renderer import GraphRenderer, GraphTrainingObserver


@dataclass(frozen=True)
class RunConfig(TabularConfig):
    environment_id: str = "RLToy/BranchingRisk-v0"
    graph_path: Path | None = None
    seed: int | None = 0
    render: bool = False
    render_every_steps: int = 1
    render_delay_ms: int = 50


def make_environment(config: RunConfig) -> gym.Env:
    if config.graph_path is not None:
        return GraphWorldEnv.from_json(config.graph_path)
    return gym.make(config.environment_id)


def run(
    train: Callable[..., TrainingResult], config: RunConfig, title: str, lambda_config: object | None = None
) -> TrainingResult:
    env = make_environment(config)
    renderer = None
    observer = None
    if config.render:
        graph_env = env.unwrapped
        if not isinstance(graph_env, GraphWorldEnv):
            raise ValueError("--render currently supports GraphWorld environments only")
        renderer = GraphRenderer(graph_env, title)
        observer = GraphTrainingObserver(renderer, config.render_every_steps, config.render_delay_ms)
    try:
        result = train(env, lambda_config or config, config.seed, observer or (lambda _: None))
        print(f"Final mean action value: {result.q_values.max(axis=1).mean():.3f}")
        return result
    finally:
        if renderer is not None:
            renderer.close()
        env.close()
