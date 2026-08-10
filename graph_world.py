"""Generic Gymnasium environment for directed graph MDPs."""

import copy
import json
import math
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from bettermdptools.algorithms.planner import Planner


def _validate_graph_spec(spec: dict) -> None:
    states = spec["states"]
    actions = spec["actions"]
    action_names = list(actions)
    start = spec["start_state"]

    action_count = len(action_names)
    if action_count == 0:
        raise ValueError("At least one action must be declared")
    if start not in states:
        raise ValueError(f"start_state '{start}' not found in states")

    for name, state in states.items():
        pos = state.get("position")
        if pos is None:
            raise ValueError(f"State '{name}' missing position")
        if not (
            isinstance(pos, (list, tuple))
            and len(pos) == 2
            and all(isinstance(v, (int, float)) and math.isfinite(v) for v in pos)
        ):
            raise ValueError(f"State '{name}' position must be [x, y] with finite floats")

        is_terminal = state.get("terminal", False)
        state_actions = state.get("actions", {})
        if is_terminal:
            if state_actions:
                raise ValueError(f"Terminal state '{name}' must not declare actions")
        else:
            for a_name in action_names:
                if a_name not in state_actions:
                    raise ValueError(
                        f"Nonterminal state '{name}' missing action '{a_name}'"
                    )
            for a_name, outcomes in state_actions.items():
                if a_name not in actions:
                    raise ValueError(
                        f"State '{name}' declares unknown action '{a_name}'"
                    )
                if not isinstance(outcomes, list) or len(outcomes) == 0:
                    raise ValueError(
                        f"State '{name}' action '{a_name}' outcomes must "
                        "be a non-empty list"
                    )
                total_p = 0.0
                for i, outcome in enumerate(outcomes):
                    p = outcome.get("probability")
                    ns = outcome.get("next_state")
                    r = outcome.get("reward", 0.0)
                    if not isinstance(p, (int, float)):
                        raise ValueError(
                            f"State '{name}' action '{a_name}' outcome {i} "
                            "missing probability"
                        )
                    if not (0.0 <= p <= 1.0):
                        raise ValueError(
                            f"State '{name}' action '{a_name}' outcome {i} "
                            f"probability {p} out of range [0,1]"
                        )
                    if ns not in states:
                        raise ValueError(
                            f"State '{name}' action '{a_name}' outcome {i} "
                            f"next_state '{ns}' not found"
                        )
                    if not math.isfinite(r):
                        raise ValueError(
                            f"State '{name}' action '{a_name}' outcome {i} "
                            f"reward {r} must be finite"
                        )
                    total_p += p
                if not math.isclose(total_p, 1.0):
                    raise ValueError(
                        f"State '{name}' action '{a_name}' probabilities "
                        f"sum to {total_p}, expected 1.0"
                    )


def _compile_P(spec: dict) -> dict:
    """Compile graph spec into the dense P[state][action] transition table."""
    states = spec["states"]
    actions = spec["actions"]
    action_names = list(actions)
    state_names = list(states)
    n_states = len(state_names)
    n_actions = len(action_names)
    state_index = {name: i for i, name in enumerate(state_names)}
    action_index = {name: i for i, name in enumerate(action_names)}

    P: dict[int, dict[int, list]] = {
        s: {a: [] for a in range(n_actions)} for s in range(n_states)
    }

    for s_name, state in states.items():
        si = state_index[s_name]
        is_terminal = state.get("terminal", False)

        if is_terminal:
            for a in range(n_actions):
                P[si][a] = [(1.0, si, 0.0, True)]
        else:
            for a_name, outcomes in state["actions"].items():
                ai = action_index[a_name]
                P[si][ai] = [
                    (
                        float(outcome["probability"]),
                        state_index[outcome["next_state"]],
                        float(outcome.get("reward", 0.0)),
                        outcome.get("terminated", False),
                    )
                    for outcome in outcomes
                ]

    return P


def load_graph_spec(path: str | Path) -> dict:
    """Load and validate a graph specification from a JSON file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Graph spec not found: {path}")
    spec = json.loads(path.read_text())
    _validate_graph_spec(spec)
    return spec


def make_graph_world(
    graph_path: str | Path,
    gamma: float = 0.99,
    step_cost: float | None = None,
):
    """Create a GraphWorldEnv and return (env, planner_values)."""
    spec = load_graph_spec(graph_path)
    env = GraphWorldEnv(spec)
    if step_cost is not None:
        env.P = _penalize_steps(env.P, step_cost)
    V, _, _ = Planner(env.P).value_iteration(gamma=gamma)
    return env, np.array(V, dtype=np.float64)


def _penalize_steps(P, cost):
    P = copy.deepcopy(P)
    for actions in P.values():
        for transitions in actions.values():
            for i, (prob, ns, r, done) in enumerate(transitions):
                if not done:
                    transitions[i] = (prob, ns, r + cost, done)
    return P


class GraphWorldEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, spec: dict):
        super().__init__()
        self._spec = spec
        self._state_names = list(spec["states"])
        self._action_names = list(spec["actions"])
        self.n_states = len(self._state_names)
        self.n_actions = len(self._action_names)
        self._state_index = {name: i for i, name in enumerate(self._state_names)}

        self._start = self._state_index[spec["start_state"]]
        self.observation_space = gym.spaces.Discrete(self.n_states)
        self.action_space = gym.spaces.Discrete(self.n_actions)

        self.P = _compile_P(spec)
        self._state = self._start

    @property
    def spec(self) -> dict:
        return self._spec

    @property
    def state_names(self) -> list[str]:
        return self._state_names

    @property
    def action_names(self) -> list[str]:
        return self._action_names

    def state_index(self, name: str) -> int:
        return self._state_index[name]

    def action_index(self, name: str) -> int:
        return self._action_names.index(name)

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple:
        super().reset(seed=seed)
        self._state = self._start
        return int(self._state), {}

    def step(self, action: int) -> tuple:
        assert self.action_space.contains(action), f"Invalid action {action}"
        outcomes = self.P[self._state][action]
        probs = [o[0] for o in outcomes]
        idx = self.np_random.choice(len(outcomes), p=probs)
        prob, next_state, reward, terminated = outcomes[idx]
        self._state = next_state
        return int(next_state), float(reward), bool(terminated), False, {}

    def terminal_states(self) -> np.ndarray:
        """Return boolean mask of terminal states."""
        terminal = np.zeros(self.n_states, dtype=bool)
        for name, state_def in self._spec["states"].items():
            if state_def.get("terminal", False):
                terminal[self._state_index[name]] = True
        return terminal