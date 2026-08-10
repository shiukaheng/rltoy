"""Pygame display for GraphWorldEnv; it is not part of the environment API."""

from collections import defaultdict
from typing import Sequence

import numpy as np
import pygame

from rltoy.algorithms.tabular import TrainingSnapshot
from rltoy.envs.graph_world import GraphWorldEnv


class GraphRenderer:
    def __init__(self, env: GraphWorldEnv, caption: str = "GraphWorld", size: tuple[int, int] = (1100, 700)):
        spec = env.graph_spec
        missing_positions = [name for name, state in spec["states"].items() if "position" not in state]
        if missing_positions:
            raise ValueError(f"Graph rendering requires positions; missing: {missing_positions}")
        self._env = env
        self._spec = spec
        self._size = size
        self._values: np.ndarray | None = None
        self._policy: np.ndarray | None = None
        self._trajectory: tuple[int, ...] = ()
        pygame.init()
        self._screen = pygame.display.set_mode(size)
        pygame.display.set_caption(caption)
        self._small = pygame.font.SysFont("monospace", 14)
        self._label = pygame.font.SysFont("monospace", 16, bold=True)
        self._value = pygame.font.SysFont("monospace", 15)

    def set_values(self, values: Sequence[float]) -> None:
        self._values = np.asarray(values, dtype=float)

    def set_policy(self, policy: Sequence[int]) -> None:
        self._policy = np.asarray(policy, dtype=int)

    def set_trajectory(self, trajectory: Sequence[int]) -> None:
        self._trajectory = tuple(int(state) for state in trajectory)

    def _position(self, state_name: str) -> tuple[int, int]:
        x, y = self._spec["states"][state_name]["position"]
        return int(70 + x * (self._size[0] - 140)), int(70 + y * (self._size[1] - 140))

    def _draw_text(self, text: str, position: tuple[int, int], font, color=(235, 235, 235)) -> None:
        image = font.render(text, True, color)
        self._screen.blit(image, image.get_rect(center=position))

    def _node_color(self, state_name: str, index: int) -> tuple[int, int, int]:
        state = self._spec["states"][state_name]
        terminal_color = state.get("display", {}).get("color")
        if terminal_color == "success":
            return (60, 175, 85)
        if terminal_color == "failure":
            return (200, 70, 70)
        if self._values is None or state.get("terminal", False):
            return (65, 105, 180)
        finite = self._values[np.isfinite(self._values)]
        if not finite.size:
            return (65, 105, 180)
        span = finite.max() - finite.min()
        intensity = 0.5 if span == 0 else (self._values[index] - finite.min()) / span
        return (int(50 + 160 * intensity), int(80 + 100 * intensity), 200)

    def render(self, current_state: int) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
        self._screen.fill((20, 22, 28))
        state_names = self._env.state_names
        action_names = self._env.action_names
        trail = set(zip(self._trajectory, self._trajectory[1:]))
        edge_counts: dict[tuple[str, str], int] = defaultdict(int)
        for source, state in self._spec["states"].items():
            for action, outcomes in state.get("actions", {}).items():
                for outcome in outcomes:
                    edge_counts[source, outcome["next_state"]] += 1

        for source, state in self._spec["states"].items():
            for action, outcomes in state.get("actions", {}).items():
                for outcome in outcomes:
                    destination = outcome["next_state"]
                    source_id = self._env.state_index(source)
                    destination_id = self._env.state_index(destination)
                    color = (255, 210, 60) if (source_id, destination_id) in trail else (145, 145, 150)
                    pygame.draw.line(self._screen, color, self._position(source), self._position(destination), 3)
                    midpoint = tuple((a + b) // 2 for a, b in zip(self._position(source), self._position(destination)))
                    label = f"{action_names[self._env.action_index(action)]} r={outcome.get('reward', 0.0):+.0f}"
                    if outcome["probability"] != 1.0:
                        label += f" p={outcome['probability']:.2f}"
                    self._draw_text(label, midpoint, self._small, color)

        for index, state_name in enumerate(state_names):
            state = self._spec["states"][state_name]
            position = self._position(state_name)
            pygame.draw.circle(self._screen, self._node_color(state_name, index), position, 42)
            pygame.draw.circle(self._screen, (240, 240, 240), position, 42, 2)
            if index == current_state:
                pygame.draw.circle(self._screen, (255, 255, 255), position, 49, 2)
            self._draw_text(state.get("label", state_name), (position[0], position[1] - 58), self._label)
            if self._values is not None and not state.get("terminal", False):
                self._draw_text(f"V={self._values[index]:.2f}", position, self._value)
            if self._policy is not None and not state.get("terminal", False):
                action = self._policy[index]
                self._draw_text(f"pi={action_names[action]}", (position[0], position[1] + 58), self._value, (255, 210, 60))
        pygame.display.flip()

    def close(self) -> None:
        pygame.quit()


class GraphTrainingObserver:
    """Render learner snapshots without coupling Pygame to the learner."""

    def __init__(self, renderer: GraphRenderer, every_steps: int = 1, delay_ms: int = 50):
        self._renderer = renderer
        self._every_steps = every_steps
        self._delay_ms = delay_ms
        self._steps = 0

    def __call__(self, snapshot: TrainingSnapshot) -> None:
        self._steps += 1
        if self._steps % self._every_steps:
            return
        self._renderer.set_values(snapshot.q_values.max(axis=1))
        self._renderer.set_policy(snapshot.q_values.argmax(axis=1))
        self._renderer.set_trajectory(snapshot.trajectory)
        self._renderer.render(snapshot.state)
        pygame.time.wait(self._delay_ms)
