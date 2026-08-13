"""A minimal PuckWorld exercise with a random policy baseline."""

import argparse
import time
from collections.abc import Callable

import gymnasium as gym
import numpy as np
import pygame
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets


class PuckWorld(gym.Env):
    """Move a blue puck to successive green targets in a small 2D world."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode: str | None = None, max_steps: int = 250):
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.action_space = gym.spaces.Discrete(5)  # Left, right, up, down, coast.
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        self.position = np.zeros(2)
        self.velocity = np.zeros(2)
        self.target = np.zeros(2)
        self.steps = 0
        self.screen: pygame.Surface | None = None
        self.clock: pygame.time.Clock | None = None

    def observation(self) -> np.ndarray:
        return np.array([*self.position, *self.velocity, *self.target], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        del options
        self.position = self.np_random.uniform(-0.8, 0.8, size=2)
        self.velocity = np.zeros(2)
        self.target = self.np_random.uniform(-0.8, 0.8, size=2)
        self.steps = 0
        return self.observation(), {}

    def step(self, action: int):
        acceleration = np.array(
            [(0.0, 0.0), (-0.04, 0.0), (0.04, 0.0), (0.0, -0.04), (0.0, 0.04)]
        )[action]
        self.velocity = np.clip(self.velocity * 0.95 + acceleration, -0.08, 0.08)
        self.position = np.clip(self.position + self.velocity, -1.0, 1.0)
        self.steps += 1

        reached_target = np.linalg.norm(self.position - self.target) < 0.1
        reward = 1.0 if reached_target else -0.01
        if reached_target:
            self.target = self.np_random.uniform(-0.8, 0.8, size=2)

        if self.render_mode == "human":
            self.render()
        return self.observation(), reward, False, self.steps >= self.max_steps, {}

    def render(self) -> None:
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((600, 600))
            pygame.display.set_caption("PuckWorld")
            self.clock = pygame.time.Clock()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt

        def point(location: np.ndarray) -> tuple[int, int]:
            return tuple(((location + 1.0) * 300).astype(int))

        self.screen.fill((20, 24, 36))
        pygame.draw.circle(self.screen, (80, 220, 120), point(self.target), 20)
        pygame.draw.circle(self.screen, (70, 160, 255), point(self.position), 16)
        pygame.display.flip()
        self.clock.tick(60)

    def close(self) -> None:
        if self.screen is not None:
            pygame.quit()
            self.screen = None


# === YOUR POLICY STARTS HERE ===
# Replace this random baseline with your action-selection rule. For SARSA, this
# is where an epsilon-greedy choice from your discretized Q-table would go.
def choose_action(observation: np.ndarray, action_count: int, rng: np.random.Generator) -> int:
    """Choose an action. Replace this random baseline with your policy."""
    del observation
    return int(rng.integers(action_count))


def run_episode(
    env: PuckWorld,
    rng: np.random.Generator,
    seed: int | None = None,
    on_step: Callable[[int, float], None] | None = None,
) -> float:
    observation, _ = env.reset(seed=seed)
    episode_return = 0.0
    step = 0

    while True:
        # === YOUR LEARNING LOOP ===
        # This transition contains everything needed for SARSA. Add state
        # discretization, Q-table updates, and extra bookkeeping here.
        action = choose_action(observation, env.action_space.n, rng)
        next_observation, reward, terminated, truncated, _ = env.step(action)
        episode_return += reward
        if on_step is not None:
            on_step(step, episode_return)
        step += 1
        if terminated or truncated:
            return episode_return
        observation = next_observation


def train(
    episodes: int, rng: np.random.Generator, seed: int | None, steps_per_episode: int
) -> np.ndarray:
    """Run episodes here; add policy setup such as a Q-table before this loop."""
    env = PuckWorld(max_steps=steps_per_episode)
    try:
        returns = np.empty(episodes)
        for episode in range(episodes):
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
    steps_per_episode: int,
) -> None:
    # === SAFE TO IGNORE ===
    # This is display code only. It calls the same policy as training and
    # continuously restarts after an episode so you can watch it. The plot
    # shows cumulative reward within each episode.
    app, plot, curve = make_reward_plot("PuckWorld Cumulative Reward")
    env = PuckWorld(render_mode="human", max_steps=steps_per_episode)
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
    # This is command-line and reporting code. Focus on choose_action,
    # run_episode, and policy state created in train.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps-per-episode", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render-episodes", type=int, default=1)
    parser.add_argument("--no-loop", action="store_true")
    parser.add_argument("--terminal-pause-seconds", type=float, default=1.0)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    if args.steps_per_episode < 1:
        parser.error("--steps-per-episode must be at least 1")
    if args.render_episodes < 0:
        parser.error("--render-episodes must be non-negative")
    if args.terminal_pause_seconds < 0:
        parser.error("--terminal-pause-seconds must be non-negative")

    rng = np.random.default_rng(args.seed)
    returns = train(args.episodes, rng, args.seed, args.steps_per_episode)
    print(f"Mean return: {returns.mean():.2f}; final return: {returns[-1]:.2f}")
    if args.render_episodes:
        show_policy(
            args.render_episodes,
            rng,
            args.seed,
            loop=not args.no_loop,
            terminal_pause_seconds=args.terminal_pause_seconds,
            steps_per_episode=args.steps_per_episode,
        )


if __name__ == "__main__":
    main()
