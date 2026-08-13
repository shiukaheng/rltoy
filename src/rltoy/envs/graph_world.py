"""A JSON-authored, Gymnasium-compatible finite graph MDP."""

import copy
import json
import math
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from bettermdptools.algorithms.planner import Planner


def _validate_graph_spec(spec: dict[str, Any]) -> None:
    """Validate the MDP fields of a graph spec; display metadata is optional."""
    required = {"start_state", "actions", "states"}
    missing = required - set(spec)
    if missing:
        raise ValueError(f"Graph spec missing required fields: {sorted(missing)}")

    states = spec["states"]
    actions = spec["actions"]
    if not isinstance(states, dict) or not states:
        raise ValueError("states must be a non-empty object")
    if not isinstance(actions, dict) or not actions:
        raise ValueError("actions must be a non-empty object")
    if spec["start_state"] not in states:
        raise ValueError(f"start_state '{spec['start_state']}' not found in states")

    for state_name, state in states.items():
        if not isinstance(state, dict):
            raise ValueError(f"State '{state_name}' must be an object")
        state_actions = state.get("actions", {})
        if state.get("terminal", False):
            if state_actions:
                raise ValueError(f"Terminal state '{state_name}' must not declare actions")
            continue
        if not isinstance(state_actions, dict):
            raise ValueError(f"State '{state_name}' actions must be an object")
        for action_name in actions:
            if action_name not in state_actions:
                raise ValueError(f"Nonterminal state '{state_name}' missing action '{action_name}'")
        for action_name, outcomes in state_actions.items():
            if action_name not in actions:
                raise ValueError(f"State '{state_name}' declares unknown action '{action_name}'")
            if not isinstance(outcomes, list) or not outcomes:
                raise ValueError(f"State '{state_name}' action '{action_name}' needs outcomes")
            probability_sum = 0.0
            for index, outcome in enumerate(outcomes):
                if not isinstance(outcome, dict):
                    raise ValueError(f"Outcome {index} for '{state_name}' must be an object")
                probability = outcome.get("probability")
                next_state = outcome.get("next_state")
                reward = outcome.get("reward", 0.0)
                if not isinstance(probability, (int, float)) or not 0.0 <= probability <= 1.0:
                    raise ValueError(f"Outcome {index} for '{state_name}' has invalid probability")
                if next_state not in states:
                    raise ValueError(f"Outcome {index} for '{state_name}' has unknown next_state")
                if not isinstance(reward, (int, float)) or not math.isfinite(reward):
                    raise ValueError(f"Outcome {index} for '{state_name}' has invalid reward")
                probability_sum += probability
            if not math.isclose(probability_sum, 1.0):
                raise ValueError(
                    f"Probabilities for '{state_name}' action '{action_name}' sum to {probability_sum}"
                )


def _compile_transitions(spec: dict[str, Any]) -> dict[int, dict[int, list[tuple]]]:
    state_names = list(spec["states"])
    action_names = list(spec["actions"])
    state_index = {name: index for index, name in enumerate(state_names)}
    action_index = {name: index for index, name in enumerate(action_names)}
    transitions = {
        state: {action: [] for action in range(len(action_names))}
        for state in range(len(state_names))
    }

    for state_name, state in spec["states"].items():
        state_id = state_index[state_name]
        if state.get("terminal", False):
            for action in range(len(action_names)):
                transitions[state_id][action] = [(1.0, state_id, 0.0, True)]
            continue
        for action_name, outcomes in state["actions"].items():
            action_id = action_index[action_name]
            transitions[state_id][action_id] = [
                (
                    float(outcome["probability"]),
                    state_index[outcome["next_state"]],
                    float(outcome.get("reward", 0.0)),
                    bool(
                        outcome.get(
                            "terminated",
                            spec["states"][outcome["next_state"]].get("terminal", False),
                        )
                    ),
                )
                for outcome in outcomes
            ]
    return transitions


def load_graph_spec(path: str | Path) -> dict[str, Any]:
    """Load and validate a graph MDP JSON file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Graph spec not found: {path}")
    spec = json.loads(path.read_text())
    _validate_graph_spec(spec)
    return spec


class GraphWorldEnv(gym.Env[int, int]):
    """Finite graph MDP with discrete observations and actions."""

    metadata = {"render_modes": []}

    def __init__(self, spec: dict[str, Any]):
        _validate_graph_spec(spec)
        self._spec = copy.deepcopy(spec)
        self._state_names = list(spec["states"])
        self._action_names = list(spec["actions"])
        self._state_index = {name: index for index, name in enumerate(self._state_names)}
        self._action_index = {name: index for index, name in enumerate(self._action_names)}
        self.observation_space = gym.spaces.Discrete(len(self._state_names))
        self.action_space = gym.spaces.Discrete(len(self._action_names))
        self.P = _compile_transitions(spec)  # Useful for exact tabular planning.
        self._start = self._state_index[spec["start_state"]]
        self._state = self._start

    @classmethod
    def from_json(cls, path: str | Path) -> "GraphWorldEnv":
        return cls(load_graph_spec(path))

    @property
    def graph_spec(self) -> dict[str, Any]:
        return copy.deepcopy(self._spec)

    @property
    def state_names(self) -> list[str]:
        return self._state_names.copy()

    @property
    def action_names(self) -> list[str]:
        return self._action_names.copy()

    def state_index(self, name: str) -> int:
        return self._state_index[name]

    def action_index(self, name: str) -> int:
        return self._action_index[name]

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[int, dict]:
        super().reset(seed=seed)
        self._state = self._start
        return self._state, {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")
        outcomes = self.P[self._state][action]
        outcome_id = self.np_random.choice(len(outcomes), p=[outcome[0] for outcome in outcomes])
        _, next_state, reward, terminated = outcomes[outcome_id]
        self._state = next_state
        return next_state, reward, terminated, False, {}

    def terminal_states(self) -> np.ndarray:
        return np.array(
            [state.get("terminal", False) for state in self._spec["states"].values()], dtype=bool
        )


def planner_values(env: GraphWorldEnv, gamma: float) -> np.ndarray:
    """Return optimal state values for a graph environment's explicit transition table."""
    values, _, _ = Planner(env.P).value_iteration(gamma=gamma)
    return np.asarray(values, dtype=np.float64)


def make_branching_risk(**_: Any) -> GraphWorldEnv:
    """Gymnasium factory for the bundled Branching Risk environment."""
    return GraphWorldEnv.from_json(Path(__file__).with_name("data") / "two_strings.json")
