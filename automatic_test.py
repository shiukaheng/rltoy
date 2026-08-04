import gymnasium as gym
from bettermdptools.algorithms.planner import Planner

from map_renderer import GridMapRenderer

env = gym.make("FrozenLake-v1", is_slippery=False)

renderer = GridMapRenderer(env.unwrapped, caption="FrozenLake (automatic)")
V, _, _ = Planner(env.unwrapped.P).value_iteration()
renderer.set_values(V)

obs, info = env.reset()
renderer.render(obs)

terminated = truncated = False

while not (terminated or truncated):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    renderer.render(obs)

renderer.close()
env.close()