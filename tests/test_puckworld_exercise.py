import numpy as np

from rltoy.exercises.puckworld import PuckWorld, choose_action, train


def test_random_puckworld_baseline_runs_requested_episodes():
    returns = train(3, np.random.default_rng(0), seed=0, steps_per_episode=5)

    assert returns.shape == (3,)
    assert np.isfinite(returns).all()


def test_puckworld_observation_and_random_action_are_valid():
    env = PuckWorld(max_steps=1)
    try:
        observation, _ = env.reset(seed=0)
        action = choose_action(observation, env.action_space.n, np.random.default_rng(0))

        assert env.observation_space.contains(observation)
        assert env.action_space.contains(action)
    finally:
        env.close()
