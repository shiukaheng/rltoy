import numpy as np

from rltoy.algorithms.deep_q_learning import DeepQConfig, ReplayBuffer, train
from rltoy.envs import GraphWorldEnv


def test_replay_buffer_samples_transitions_without_replacement():
    replay = ReplayBuffer(2)
    replay.add(0, 0, 1.0, 1, False)
    replay.add(1, 1, 2.0, 2, True)

    states, actions, rewards, next_states, dones = replay.sample(2, np.random.default_rng(0))

    assert set(states) == {0, 1}
    assert set(actions) == {0, 1}
    assert set(rewards) == {1.0, 2.0}
    assert set(next_states) == {1, 2}
    assert set(dones) == {False, True}


def test_deep_q_learning_runs_on_a_one_step_environment():
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
    config = DeepQConfig(episodes=3, batch_size=1, epsilon_initial=0.0, epsilon_final=0.0)

    result = train(env, config, seed=0)

    assert result.q_values.shape == (2, 1)
    assert result.state_values.shape == (3, 2)
    assert result.episode_returns.tolist() == [3.0, 3.0, 3.0]
    assert np.isfinite(result.q_values).all()
