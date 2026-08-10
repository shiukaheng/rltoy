"""Explore a GraphWorld manually using action keys from its JSON display metadata."""

import pygame

from rltoy.envs.graph_world import make_branching_risk, planner_values
from rltoy.visualization.graph_renderer import GraphRenderer


def main() -> None:
    env = make_branching_risk()
    renderer = GraphRenderer(env, "Branching Risk (interactive)")
    renderer.set_state_values(planner_values(env, gamma=0.99))
    keys = {
        getattr(pygame, f"K_{details['key']}"): env.action_index(name)
        for name, details in env.graph_spec["actions"].items()
        if "key" in details
    }
    state, _ = env.reset()
    trajectory = [state]
    renderer.set_trajectory(trajectory)
    renderer.render(state)
    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type != pygame.KEYDOWN:
                    continue
                if event.key == pygame.K_r:
                    state, _ = env.reset()
                    trajectory = [state]
                elif event.key in keys:
                    state, reward, terminated, _, _ = env.step(keys[event.key])
                    trajectory.append(state)
                    print(f"reward: {reward:+.1f}")
                    if terminated:
                        renderer.set_trajectory(trajectory)
                        renderer.render(state)
                        pygame.time.wait(800)
                        state, _ = env.reset()
                        trajectory = [state]
                renderer.set_trajectory(trajectory)
                renderer.render(state)
    finally:
        renderer.close()
        env.close()
