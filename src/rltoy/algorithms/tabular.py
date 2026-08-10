"""Small mechanics shared by tabular algorithms, not their update rules."""

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class TabularConfig:
    episodes: int = 1_000
    gamma: float = 0.99
    alpha_initial: float = 0.5
    alpha_final: float = 0.01
    alpha_decay_ratio: float = 0.5
    epsilon_initial: float = 1.0
    epsilon_final: float = 0.1
    epsilon_decay_ratio: float = 0.9


@dataclass(frozen=True)
class TrainingSnapshot:
    episode: int
    state: int
    q_values: np.ndarray
    trajectory: tuple[int, ...]


@dataclass(frozen=True)
class TrainingResult:
    q_values: np.ndarray
    episode_returns: np.ndarray
    state_values: np.ndarray


Observer = Callable[[TrainingSnapshot], None]


def observe_nothing(_: TrainingSnapshot) -> None:
    """Default observer, so learners need no visualization conditionals."""


def decay_schedule(initial: float, final: float, ratio: float, steps: int) -> np.ndarray:
    if steps < 1:
        raise ValueError("episodes must be at least 1")
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("decay ratio must be between 0 and 1")
    if steps == 1:
        return np.array([initial])
    decay_steps = min(steps, max(2, int(steps * ratio)))
    fraction = np.linspace(0.0, 1.0, decay_steps)
    values = initial + fraction * (final - initial)
    return np.pad(values, (0, steps - decay_steps), mode="edge")


def epsilon_greedy(q_values: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(len(q_values)))
    return int(rng.choice(np.flatnonzero(q_values == q_values.max())))
