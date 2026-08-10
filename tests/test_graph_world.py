import json

import gymnasium as gym
import numpy as np
import pytest

import rltoy
from rltoy.envs import GraphWorldEnv
from rltoy.envs.graph_world import _validate_graph_spec, planner_values


@pytest.fixture
def one_step_spec():
    return {
        "start_state": "start",
        "actions": {"left": {}, "right": {}},
        "states": {
            "start": {
                "actions": {
                    "left": [{"probability": 1.0, "next_state": "goal", "reward": 2.0}],
                    "right": [{"probability": 1.0, "next_state": "loss", "reward": -1.0}],
                }
            },
            "goal": {"terminal": True},
            "loss": {"terminal": True},
        },
    }


def test_graph_world_is_a_gymnasium_environment(one_step_spec):
    env = GraphWorldEnv(one_step_spec)
    assert isinstance(env, gym.Env)
    state, _ = env.reset(seed=3)
    next_state, reward, terminated, truncated, _ = env.step(0)
    assert (state, next_state, reward, terminated, truncated) == (0, 1, 2.0, True, False)
    assert env.P[1][0] == [(1.0, 1, 0.0, True)]


def test_display_metadata_is_optional(one_step_spec):
    _validate_graph_spec(one_step_spec)


def test_invalid_probability_is_rejected(one_step_spec):
    one_step_spec["states"]["start"]["actions"]["left"][0]["probability"] = 0.7
    with pytest.raises(ValueError, match="sum"):
        _validate_graph_spec(one_step_spec)


def test_json_loading_and_planner(tmp_path, one_step_spec):
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(one_step_spec))
    env = GraphWorldEnv.from_json(path)
    values = planner_values(env, gamma=0.99)
    assert values[env.state_index("start")] == pytest.approx(2.0)


def test_bundled_graph_is_registered_with_gymnasium():
    env = gym.make("RLToy/BranchingRisk-v0")
    try:
        assert env.observation_space.n == 10
        assert env.action_space.n == 3
        assert np.all(env.unwrapped.terminal_states()[-5:])
        state, _ = env.reset(seed=0)
        assert state == env.unwrapped.state_index("start")
    finally:
        env.close()
