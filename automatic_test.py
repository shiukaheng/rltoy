import time

from frozen_lake import make_frozen_lake
from map_renderer import GridMapRenderer

DELAY_S = 0.3

env, V = make_frozen_lake(is_slippery=False)

renderer = GridMapRenderer(env.unwrapped, caption="FrozenLake (automatic)")
renderer.set_values(V)

obs, info = env.reset()
renderer.render(obs)

terminated = truncated = False

while not (terminated or truncated):
    time.sleep(DELAY_S)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    renderer.render(obs)

renderer.close()
env.close()