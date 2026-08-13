"""Pygame display for GraphWorldEnv; it is not part of the environment API."""

from collections import defaultdict
from typing import Sequence

import numpy as np
import pygame

from rltoy.algorithms.tabular import TrainingSnapshot
from rltoy.envs.graph_world import GraphWorldEnv


LAVA_COLORS = (
    (22, 18, 46),
    (82, 29, 91),
    (165, 47, 92),
    (230, 83, 56),
    (252, 164, 54),
    (252, 255, 164),
)


def lava_color(intensity: float) -> tuple[int, int, int]:
    """Interpolate the dark-purple-to-yellow lava palette at an intensity in [0, 1]."""
    intensity = float(np.clip(intensity, 0.0, 1.0))
    position = intensity * (len(LAVA_COLORS) - 1)
    lower = int(position)
    upper = min(lower + 1, len(LAVA_COLORS) - 1)
    fraction = position - lower
    return tuple(
        round(start + fraction * (end - start))
        for start, end in zip(LAVA_COLORS[lower], LAVA_COLORS[upper], strict=True)
    )


class GraphRenderer:
    def __init__(self, env: GraphWorldEnv, caption: str = "GraphWorld", size: tuple[int, int] = (1100, 700)):
        spec = env.graph_spec
        missing_positions = [name for name, state in spec["states"].items() if "position" not in state]
        if missing_positions:
            raise ValueError(f"Graph rendering requires positions; missing: {missing_positions}")
        self._env = env
        self._spec = spec
        self._size = size
        self._state_values: np.ndarray | None = None
        self._policy: np.ndarray | None = None
        self._policy_probabilities: np.ndarray | None = None
        self._trajectory: tuple[int, ...] = ()
        pygame.init()
        self._screen = pygame.display.set_mode(size)
        pygame.display.set_caption(caption)
        self._small = pygame.font.SysFont("monospace", 14)
        self._label = pygame.font.SysFont("monospace", 16, bold=True)
        self._value = pygame.font.SysFont("monospace", 15)

    def set_state_values(self, values: Sequence[float]) -> None:
        """Display estimated state values, used by action-value learners and planners."""
        self._state_values = np.asarray(values, dtype=float)

    def set_policy_probabilities(self, probabilities: np.ndarray) -> None:
        """Display an action policy on its corresponding outgoing graph edges."""
        self._policy_probabilities = np.asarray(probabilities, dtype=float)
        self._policy = None

    def set_policy(self, policy: Sequence[int]) -> None:
        self._policy = np.asarray(policy, dtype=int)

    def set_trajectory(self, trajectory: Sequence[int]) -> None:
        self._trajectory = tuple(int(state) for state in trajectory)

    def _position(self, state_name: str) -> tuple[int, int]:
        x, y = self._spec["states"][state_name]["position"]
        return int(70 + x * (self._size[0] - 140)), int(70 + y * (self._size[1] - 140))

    def _draw_text(self, text: str, position: tuple[int, int], font, color=(235, 235, 235), outline: tuple[int, int, int] | None = (20, 22, 28)) -> None:
        image = font.render(text, True, color)
        rect = image.get_rect(center=position)
        if outline is not None:
            outline_image = font.render(text, True, outline)
            for dx, dy in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
                self._screen.blit(outline_image, (rect.x + dx, rect.y + dy))
        self._screen.blit(image, rect)

    def _node_color(self, state_name: str, index: int) -> tuple[int, int, int]:
        state = self._spec["states"][state_name]
        terminal_color = state.get("display", {}).get("color")
        if terminal_color == "success":
            return (60, 175, 85)
        if terminal_color == "failure":
            return (200, 70, 70)
        if self._state_values is None or state.get("terminal", False):
            return (65, 105, 180)
        finite = self._state_values[np.isfinite(self._state_values)]
        if not finite.size:
            return (65, 105, 180)
        span = finite.max() - finite.min()
        intensity = 0.5 if span == 0 else (self._state_values[index] - finite.min()) / span
        return lava_color(intensity)

    def _edge_style(self, probability: float | None, on_trajectory: bool) -> tuple[tuple[int, int, int], int]:
        if on_trajectory:
            return (80, 220, 255), 4
        if probability is None:
            return (145, 145, 150), 3
        return lava_color(0.15 + 0.85 * probability), int(2 + 4 * probability)

    def _draw_color_key(self) -> None:
        if self._state_values is None and self._policy_probabilities is None:
            return
        if self._state_values is not None and self._policy_probabilities is not None:
            label = "V nodes | pi edges: low -> high"
        elif self._state_values is not None:
            label = "V: low -> high"
        else:
            label = "pi: low -> high"
        origin = (24, self._size[1] - 36)
        width = 18
        for index in range(len(LAVA_COLORS)):
            pygame.draw.rect(
                self._screen,
                LAVA_COLORS[index],
                (origin[0] + index * width, origin[1], width, 12),
            )
        self._draw_text(label, (origin[0] + 118, origin[1] + 6), self._small)

    def render(self, current_state: int) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
        self._screen.fill((20, 22, 28))
        self._draw_color_key()
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
                    action_id = self._env.action_index(action)
                    policy_probability = (
                        None
                        if self._policy_probabilities is None
                        else self._policy_probabilities[source_id, action_id]
                    )
                    color, width = self._edge_style(
                        policy_probability, (source_id, destination_id) in trail
                    )
                    pygame.draw.line(self._screen, color, self._position(source), self._position(destination), width)
                    midpoint = tuple((a + b) // 2 for a, b in zip(self._position(source), self._position(destination)))
                    label = f"{action_names[action_id]} r={outcome.get('reward', 0.0):+.0f}"
                    if outcome["probability"] != 1.0:
                        label += f" p={outcome['probability']:.2f}"
                    if policy_probability is not None:
                        label += f" pi={policy_probability:.2f}"
                    self._draw_text(label, midpoint, self._small, color)

        for index, state_name in enumerate(state_names):
            state = self._spec["states"][state_name]
            position = self._position(state_name)
            pygame.draw.circle(self._screen, self._node_color(state_name, index), position, 42)
            pygame.draw.circle(self._screen, (240, 240, 240), position, 42, 2)
            if index == current_state:
                pygame.draw.circle(self._screen, (255, 255, 255), position, 49, 2)
            self._draw_text(state.get("label", state_name), (position[0], position[1] - 58), self._label)
            if self._state_values is not None and not state.get("terminal", False):
                self._draw_text(f"V={self._state_values[index]:.2f}", position, self._value)
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
        if snapshot.action_values is not None:
            self._renderer.set_state_values(snapshot.action_values.max(axis=1))
            self._renderer.set_policy(snapshot.action_values.argmax(axis=1))
        if snapshot.state_values is not None:
            self._renderer.set_state_values(snapshot.state_values)
        if snapshot.action_probabilities is not None:
            self._renderer.set_policy_probabilities(snapshot.action_probabilities)
        self._renderer.set_trajectory(snapshot.trajectory)
        self._renderer.render(snapshot.state)
        pygame.time.wait(self._delay_ms)
