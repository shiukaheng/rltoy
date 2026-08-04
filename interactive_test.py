import copy
import pygame
import gymnasium as gym
from bettermdptools.algorithms.planner import Planner

from map_renderer import GridMapRenderer

STEP_COST = -0.04

def penalize_steps(P, cost):
    P = copy.deepcopy(P)
    for actions in P.values():
        for transitions in actions.values():
            for i, (prob, ns, r, done) in enumerate(transitions):
                if not done:
                    transitions[i] = (prob, ns, r + cost, done)
    return P

env = gym.make("FrozenLake-v1", is_slippery=False)

PP = penalize_steps(env.unwrapped.P, STEP_COST)

renderer = GridMapRenderer(env.unwrapped, caption="FrozenLake (interactive)")
V, _, _ = Planner(PP).value_iteration()
renderer.set_values(V)

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