"""A minimal CartPole exercise with a random policy baseline."""

import argparse
import time
from collections.abc import Callable

import gymnasium as gym
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
import random


# === YOUR POLICY STARTS HERE ===
# Replace this random baseline with your action-selection rule. For SARSA, this
# is where an epsilon-greedy choice from your Q-table would go.

"""
IMPLEMENTATION
We will do SARSA
We will quantize the state space into 6 segments
And output two discrete actions

SARSA loop:
- init Q
"""

q = np.zeros((6, 2))
EPSILON = 0.2

def run_episode(
    env: gym.Env,
    rng: np.random.Generator,
    seed: int | None = None,
    on_step: Callable[[int, float], None] | None = None,
) -> float:
    observation, _ = env.reset(seed=seed)
    episode_return = 0.0
    step = 0

    while True:
        # === YOUR LEARNING LOOP ===
        # `observation`, `action`, `reward`, and the next `observation` are the
        # transition available for a SARSA update. Add state discretization,
        # Q-table updates, and any extra bookkeeping here.

        """
        OBSERVATION PROFILE
        ndarray: (4, )
        observations: [cart pos, vel, angle,   angular vel] (episodes terminate if cart leaves +-2.4)
        domain:  [+-4.8.  , -  , +-0.418, -]
        """

        """
        ACTION PROFILE
        ndarray: (1, )
        action: [move]
        domain:  [0|1] (0=left, 1=right)
        """

        action = policy_learning(observation, env.action_space.n, rng)
        next_observation, reward, terminated, truncated, _ = env.step(action)
        episode_return += reward
        if on_step is not None:
            on_step(step, episode_return)
        step += 1
        if terminated or truncated:
            return episode_return
        observation = next_observation

def learning_policy(sn):
    if random.uniform() > EPSILON:
        return int(np.argmax(q[sn]))
    else:
        return int(np.array(random.uniform() > EPSILON))

def train(episodes: int) -> np.ndarray:
    """Run episodes here; add policy setup such as a Q-table before this loop."""
    env = gym.make("CartPole-v1")
    try:
        for episode in range(episodes):
            t = 0
            sn, _ = env.reset()
            while True:
                an = learning_policy(sn)
                snp1, rnp1, term, trunc = env.step(an)
                td_target = 
            returns[episode] = run_episode(env, rng, seed if episode == 0 else None)
        return returns
    finally:
        env.close()


def make_reward_plot(title: str):
    # === SAFE TO IGNORE ===
    # Live plot of cumulative reward within the current inference episode.
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    plot = pg.PlotWidget(title=title)
    plot.setBackground("#16122e")
    plot.setLabel("bottom", "Step")
    plot.setLabel("left", "Cumulative reward")
    plot.showGrid(x=True, y=True, alpha=0.3)
    curve = plot.plot([], [], pen=pg.mkPen("#fca434", width=2))
    plot.show()
    return app, plot, curve


def show_policy(
    episodes: int,
    rng: np.random.Generator,
    seed: int | None,
    loop: bool,
    terminal_pause_seconds: float,
) -> None:
    # === SAFE TO IGNORE ===
    # This is only display code. It calls the same policy as training and
    # continuously restarts after failure so you can watch it. No learning
    # happens here. The plot shows cumulative reward within each episode.
    """Play the final policy using Gymnasium's human renderer."""
    app, plot, curve = make_reward_plot("CartPole Cumulative Reward")
    env = gym.make("CartPole-v1", render_mode="human")
    try:
        episode = 0
        while loop or episode < episodes:
            xs: list[int] = []
            ys: list[float] = []

            def track(step: int, cumulative: float) -> None:
                xs.append(step)
                ys.append(cumulative)
                curve.setData(xs, ys)
                app.processEvents()

            run_episode(env, rng, seed if episode == 0 else None, on_step=track)
            episode += 1
            time.sleep(terminal_pause_seconds)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        plot.close()


def main() -> None:
    # === SAFE TO IGNORE ===
    # This is command-line and reporting code. Your implementation can focus
    # on choose_action, run_episode, and any policy state created in train.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render-episodes", type=int, default=1)
    parser.add_argument("--no-loop", action="store_true")
    parser.add_argument("--terminal-pause-seconds", type=float, default=1.0)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    if args.render_episodes < 0:
        parser.error("--render-episodes must be non-negative")
    if args.terminal_pause_seconds < 0:
        parser.error("--terminal-pause-seconds must be non-negative")

    rng = np.random.default_rng(args.seed)
    returns = train(args.episodes, rng, args.seed)
    print(f"Mean return: {returns.mean():.1f}; final return: {returns[-1]:.1f}")
    if args.render_episodes:
        show_policy(
            args.render_episodes,
            rng,
            args.seed,
            loop=not args.no_loop,
            terminal_pause_seconds=args.terminal_pause_seconds,
        )


if __name__ == "__main__":
    main()
