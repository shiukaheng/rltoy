import pygame

from frozen_lake import make_frozen_lake
from map_renderer import GridMapRenderer

env, V = make_frozen_lake(
    is_slippery=False,
    step_cost=-0.04,
)

renderer = GridMapRenderer(env.unwrapped, caption="FrozenLake (interactive)")
renderer.set_values(V)

obs, info = env.reset()
renderer.render(obs)

while True:
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                action = 0
            elif event.key == pygame.K_DOWN:
                action = 1
            elif event.key == pygame.K_RIGHT:
                action = 2
            elif event.key == pygame.K_UP:
                action = 3
            else:
                continue
            obs, reward, terminated, truncated, info = env.step(action)
            renderer.render(obs)
            if terminated or truncated:
                obs, info = env.reset()
                renderer.render(obs)
        elif event.type == pygame.QUIT:
            renderer.close()
            env.close()
            raise SystemExit

renderer.close()
env.close()