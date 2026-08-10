import pytest
import numpy as np

from rltoy.algorithms import q_learning, sarsa, sarsa_lambda
from rltoy.algorithms.sarsa_lambda import SarsaLambdaConfig
from rltoy.algorithms.tabular import TabularConfig, decay_schedule
from rltoy.envs import GraphWorldEnv


@pytest.fixture
def one_step_environment():
    return GraphWorldEnv(
        {
            "start_state": "start",
            "actions": {"reward": {}},
            "states": {
                "start": {"actions": {"reward": [{"probability": 1.0, "next_state": "goal", "reward": 3.0}]}},
                "goal": {"terminal": True},
            },
        }
    )


@pytest.mark.parametrize("train", [q_learning.train, sarsa.train])
def test_one_step_td_algorithms_update_to_terminal_reward(one_step_environment, train):
    config = TabularConfig(episodes=1, alpha_initial=1.0, epsilon_initial=0.0)
    result = train(one_step_environment, config, seed=0)
    assert result.q_values[0, 0] == pytest.approx(3.0)
    assert result.episode_returns.tolist() == [3.0]


def test_sarsa_lambda_updates_to_terminal_reward(one_step_environment):
    config = SarsaLambdaConfig(episodes=1, alpha_initial=1.0, epsilon_initial=0.0)
    result = sarsa_lambda.train(one_step_environment, config, seed=0)
    assert result.q_values[0, 0] == pytest.approx(3.0)


def test_schedules_reach_and_hold_final_value():
    schedule = decay_schedule(1.0, 0.2, 0.5, 10)
    assert schedule[0] == pytest.approx(1.0)
    assert schedule[4] == pytest.approx(0.2)
    assert np.allclose(schedule[5:], 0.2)
