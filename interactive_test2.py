import pygame
import gymnasium as gym

from map_renderer import GridMapRenderer

env = gym.make("FrozenLake-v1", is_slippery=False)

renderer = GridMapRenderer(env.unwrapped, caption="FrozenLake (interactive)")

obs, info = env.reset()
renderer.render(obs)

terminated = truncated = False

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