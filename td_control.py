"""Shared GraphWorld training loop for the tabular TD-control algorithms."""

from dataclasses import dataclass

import numpy as np
import pygame

from graph_renderer import GraphRenderer
from graph_world import make_graph_world


@dataclass
class Config:
    n_episodes: int = 1000
    gamma: float = 0.99
    alpha_init: float = 0.5
    alpha_min: float = 0.01
    alpha_decay_ratio: float = 0.5
    epsilon_init: float = 1.0
    epsilon_min: float = 0.1
    epsilon_decay_ratio: float = 0.9
    step_delay: float = 0.1
    evaluation_step_delay: float = 0.3
    seed: int | None = None


@dataclass
class LambdaConfig(Config):
    trace_decay: float = 0.8


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


def run(algorithm: str, config: Config) -> None:
    """Train and evaluate one algorithm on the fixed branching-risk graph."""
    env, _ = make_graph_world("graphs/branching_risk.json", gamma=config.gamma)
    renderer = GraphRenderer(env, caption=f"{algorithm} (training)")
    rng = np.random.default_rng(config.seed)
    Q = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float32)
    alphas = decay_schedule(config.alpha_init, config.alpha_min, config.alpha_decay_ratio, config.n_episodes)
    epsilons = decay_schedule(config.epsilon_init, config.epsilon_min, config.epsilon_decay_ratio, config.n_episodes)

    def choose_action(state: int, epsilon: float) -> int:
        if rng.random() < epsilon:
            return int(rng.integers(env.action_space.n))
        values = Q[state]
        return int(rng.choice(np.flatnonzero(values == values.max())))

    try:
        for episode in range(config.n_episodes):
            state, _ = env.reset(seed=config.seed if episode == 0 else None)
            trajectory = [state]
            action = choose_action(state, epsilons[episode])
            traces = np.zeros_like(Q) if algorithm == "SARSA(lambda)" else None
            terminated = False

            while not terminated:
                next_state, reward, terminated, _, _ = env.step(action)

                if algorithm == "Q-learning":
                    target = reward + config.gamma * np.max(Q[next_state]) * (not terminated)
                    Q[state, action] += alphas[episode] * (target - Q[state, action])
                    next_action = choose_action(next_state, epsilons[episode])
                else:
                    next_action = choose_action(next_state, epsilons[episode])
                    target = reward + config.gamma * Q[next_state, next_action] * (not terminated)
                    error = target - Q[state, action]
                    if algorithm == "SARSA(lambda)":
                        traces[state, action] += 1
                        Q += alphas[episode] * error * traces
                        traces *= config.gamma * config.trace_decay
                    else:
                        Q[state, action] += alphas[episode] * error

                state, action = next_state, next_action
                trajectory.append(state)
                renderer.set_values(np.max(Q, axis=1))
                renderer.set_trajectory(trajectory)
                renderer.render(state)
                pygame.display.set_caption(f"{algorithm} - episode {episode + 1}/{config.n_episodes}")
                pygame.time.wait(int(config.step_delay * 1000))

        pygame.display.set_caption(f"{algorithm} (evaluation)")
        state, _ = env.reset()
        trajectory = [state]
        renderer.set_values(np.max(Q, axis=1))
        renderer.set_trajectory(trajectory)
        renderer.render(state)
        terminated = False
        while not terminated:
            pygame.time.wait(int(config.evaluation_step_delay * 1000))
            state, _, terminated, _, _ = env.step(choose_action(state, epsilon=0.0))
            trajectory.append(state)
            renderer.set_trajectory(trajectory)
            renderer.render(state)
    except KeyboardInterrupt:
        pass
    finally:
        renderer.close()
        env.close()
