import numpy as np

from rltoy.algorithms.reinforce import ReinforceConfig, discounted_returns, train
from rltoy.envs import GraphWorldEnv


def test_discounted_returns_use_each_step_return_to_go():
    assert discounted_returns([1.0, 2.0, 3.0], gamma=0.5) == [2.75, 3.5, 3.0]


def test_reinforce_runs_on_a_one_step_environment():
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

    result = train(env, ReinforceConfig(episodes=3), seed=0)

    assert result.q_values.shape == (2, 1)
    assert result.state_values.shape == (3, 2)
    assert result.episode_returns.tolist() == [3.0, 3.0, 3.0]
    assert np.isfinite(result.q_values).all()
    assert np.allclose(result.q_values.sum(axis=1), 1.0)
