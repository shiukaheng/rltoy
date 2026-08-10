from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pygame
import tyro

from frozen_lake import make_frozen_lake
from graph_world import make_graph_world
from map_renderer import GridMapRenderer
from graph_renderer import GraphRenderer


@dataclass
class Config:
    env: str = "frozen_lake"
    graph_path: Path = Path("graphs/risk_vs_delay.json")
    n_episodes: int = 500
    gamma: float = 0.99
    alpha_init: float = 0.5
    alpha_min: float = 0.01
    alpha_decay_ratio: float = 0.5
    epsilon_init: float = 1.0
    epsilon_min: float = 0.1
    epsilon_decay_ratio: float = 0.9
    step_delay: float = 0.01
    episode_delay: float = 0.005
    evaluation_step_delay: float = 0.3
    is_slippery: bool = False
    seed: int | None = None


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


def main(config: Config) -> None:
    rng = np.random.default_rng(config.seed)
    if config.env == "frozen_lake":
        env, _ = make_frozen_lake(is_slippery=config.is_slippery)
        renderer = GridMapRenderer(env.unwrapped, caption="FrozenLake SARSA (training)")
    else:
        env, _ = make_graph_world(config.graph_path)
        renderer = GraphRenderer(env, caption="GraphWorld SARSA (training)")

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions), dtype=np.float32)
    alphas = decay_schedule(config.alpha_init, config.alpha_min, config.alpha_decay_ratio, config.n_episodes)
    epsilons = decay_schedule(config.epsilon_init, config.epsilon_min, config.epsilon_decay_ratio, config.n_episodes)

    def select_action(state: int, epsilon: float) -> int:
        if rng.random() < epsilon:
            return int(rng.integers(n_actions))
        values = Q[state]
        return int(rng.choice(np.flatnonzero(values == values.max())))

    try:
        for episode in range(config.n_episodes):
            state, _ = env.reset(seed=config.seed if episode == 0 else None)
            terminated, truncated = False, False
            action = select_action(state, epsilons[episode])

            while not (terminated or truncated):
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                next_action = select_action(next_state, epsilons[episode])
                td_target = reward + config.gamma * Q[next_state, next_action] * (not done)
                Q[state, action] += alphas[episode] * (td_target - Q[state, action])
                state, action = next_state, next_action

                renderer.set_values(np.max(Q, axis=1))
                renderer.render(state)
                pygame.display.set_caption(
                    f"SARSA - ep {episode + 1}/{config.n_episodes}"
                )
                pygame.time.wait(int(config.step_delay * 1000))

            pygame.time.wait(int(config.episode_delay * 1000))

        pygame.display.set_caption("SARSA (evaluation)")
        obs, _ = env.reset()
        renderer.set_values(np.max(Q, axis=1))
        renderer.render(obs)

        terminated, truncated = False, False
        while not (terminated or truncated):
            pygame.time.wait(int(config.evaluation_step_delay * 1000))
            action = select_action(obs, epsilon=0.0)
            obs, reward, terminated, truncated, _ = env.step(action)
            renderer.render(obs)
    except KeyboardInterrupt:
        pass
    finally:
        renderer.close()
        env.close()


if __name__ == "__main__":
    main(tyro.cli(Config))