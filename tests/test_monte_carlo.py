import numpy as np
import pytest

from rltoy.algorithms.monte_carlo import MonteCarloConfig, first_visit_returns, train
from rltoy.envs import GraphWorldEnv


def test_first_visit_returns_exclude_repeated_state_action_pairs():
    returns = first_visit_returns(
        [(0, 0, 1.0), (1, 0, 2.0), (0, 0, 3.0)], gamma=0.5
    )

    assert returns == [(0, 0, 2.75), (1, 0, 3.5)]


def test_monte_carlo_updates_to_one_step_terminal_return():
    env = GraphWorldEnv(
        {
            "start_state": "start",
            "actions": {"reward": {}},
            "states": {
                "start": {"actions": {"reward": [{"probability": 1.0, "next_state": "goal", "reward": 3.0}]}},
                "goal": {"terminal": True},
            },
        }
    )

    result = train(env, MonteCarloConfig(episodes=1, epsilon_initial=0.0), seed=0)

    assert result.q_values[0, 0] == pytest.approx(3.0)
    assert result.episode_returns.tolist() == [3.0]
    assert np.isfinite(result.q_values).all()
