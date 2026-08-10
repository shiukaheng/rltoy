import pygame

from graph_renderer import GraphRenderer
from graph_world import make_graph_world

env, values = make_graph_world("graphs/branching_risk.json", gamma=0.95)
renderer = GraphRenderer(env, caption="Branching Risk (interactive)")
renderer.set_values(values)

keys = {pygame.K_a: 0, pygame.K_b: 1, pygame.K_c: 2}
state, _ = env.reset()
renderer.render(state)

print("A: +5 now | B: two-level branch | C: three-level branch | R: reset")

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            renderer.close()
            env.close()
            raise SystemExit
        if event.type != pygame.KEYDOWN:
            continue
        if event.key == pygame.K_r:
            state, _ = env.reset()
        elif event.key in keys:
            action = keys[event.key]
            state, reward, terminated, _, _ = env.step(action)
            print(f"Action: {env.action_names[action]}, reward: {reward:+.1f}")
            if terminated:
                pygame.time.wait(800)
                state, _ = env.reset()
        else:
            continue
        renderer.render(state)
