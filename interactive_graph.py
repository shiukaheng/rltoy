import pygame

from graph_world import make_graph_world
from graph_renderer import GraphRenderer

env, V = make_graph_world("graphs/risk_vs_delay.json", gamma=0.95)

renderer = GraphRenderer(env, caption="GraphWorld (interactive)")
renderer.set_values(V)

obs, info = env.reset()
renderer.render(obs)

action_keys = {}
for name, adef in env.spec["actions"].items():
    key = adef.get("key", name[0])
    action_keys[key] = env.action_index(name)
action_keys["r"] = -1  # reset sentinel

print(f"Keys: {', '.join(f'{k}={name}' for name, adef in env.spec['actions'].items())}, R=reset")

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            renderer.close()
            env.close()
            raise SystemExit
        if event.type == pygame.KEYDOWN:
            key = event.unicode.lower()
            if key == "r":
                obs, info = env.reset()
                renderer.render(obs)
            elif key in action_keys:
                a_idx = action_keys[key]
                if a_idx == -1:
                    continue
                obs, reward, terminated, truncated, info = env.step(a_idx)
                renderer.render(obs)
                print(f"Action: {env.action_names[a_idx]}, Reward: {reward:.1f}, Terminal: {terminated}")
                if terminated or truncated:
                    pygame.time.wait(800)
                    obs, info = env.reset()
                    renderer.render(obs)