import numpy as np

from rltoy.algorithms.actor_critic import ActorCriticConfig, train
from rltoy.envs import GraphWorldEnv


def test_actor_critic_updates_on_a_one_step_environment():
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

    result = train(env, ActorCriticConfig(episodes=3), seed=0, observer=snapshots.append)

    assert result.q_values.shape == (2, 1)
    assert result.state_values.shape == (3, 2)
    assert result.episode_returns.tolist() == [3.0, 3.0, 3.0]
    assert np.isfinite(result.q_values).all()
    assert np.isfinite(result.state_values).all()
    assert np.allclose(result.q_values.sum(axis=1), 1.0)
    assert snapshots[-1].state_values is not None
    assert snapshots[-1].action_probabilities is not None
