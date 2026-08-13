import numpy as np

from rltoy.exercises.cartpole import choose_action, train


def test_random_cartpole_baseline_runs_requested_episodes():
    rng = np.random.default_rng(0)

    returns = train(3, rng, seed=0)

    assert returns.shape == (3,)
    assert np.isfinite(returns).all()
    assert (returns > 0).all()


def test_random_policy_selects_a_valid_action():
    action = choose_action(np.zeros(4), action_count=2, rng=np.random.default_rng(0))

    assert action in {0, 1}
