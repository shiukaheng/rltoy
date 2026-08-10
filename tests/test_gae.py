import numpy as np

from rltoy.algorithms.gae import GAEConfig, generalized_advantages, train
from rltoy.envs import GraphWorldEnv


def test_generalized_advantages_mix_td_errors_over_a_trajectory():
    advantages = generalized_advantages(
        rewards=[1.0, 2.0],
        values=np.array([0.5, 1.0]),
        bootstrap_value=0.0,
        gamma=0.9,
        gae_lambda=0.8,
        terminated=True,
    )

    assert np.allclose(advantages, [2.12, 1.0])


def test_generalized_advantages_bootstraps_when_an_episode_is_truncated():
    advantages = generalized_advantages(
        rewards=[2.0],
        values=np.array([1.0]),
        bootstrap_value=4.0,
        gamma=0.9,
        gae_lambda=0.95,
        terminated=False,
    )

    assert np.allclose(advantages, [4.6])


def test_gae_runs_on_a_one_step_environment():
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
    snapshots = []

    result = train(env, GAEConfig(episodes=3), seed=0, observer=snapshots.append)

    assert result.q_values.shape == (2, 1)
    assert result.state_values.shape == (3, 2)
    assert result.episode_returns.tolist() == [3.0, 3.0, 3.0]
    assert np.isfinite(result.q_values).all()
    assert np.isfinite(result.state_values).all()
    assert np.allclose(result.q_values.sum(axis=1), 1.0)
    assert snapshots[-1].state_values is not None
    assert snapshots[-1].action_probabilities is not None
