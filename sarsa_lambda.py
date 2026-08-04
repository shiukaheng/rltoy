import numpy as np
import pygame

from bettermdptools.algorithms.rl import RL
from frozen_lake import make_frozen_lake
from map_renderer import GridMapRenderer

N_EPISODES = 500
GAMMA = 0.99
LAMBDA = 0.8
STEP_DELAY = 0.01
EPISODE_DELAY = 0.005
ALPHA_INIT, ALPHA_MIN, ALPHA_RATIO = 0.5, 0.01, 0.5
EPS_INIT, EPS_MIN, EPS_RATIO = 1.0, 0.1, 0.9

env, _ = make_frozen_lake(is_slippery=False)
nS = env.observation_space.n
nA = env.action_space.n

renderer = GridMapRenderer(env.unwrapped, caption="FrozenLake SARSA(lambda) (training)")
Q = np.zeros((nS, nA), dtype=np.float32)

alphas = RL.decay_schedule(ALPHA_INIT, ALPHA_MIN, ALPHA_RATIO, N_EPISODES)
epsilons = RL.decay_schedule(EPS_INIT, EPS_MIN, EPS_RATIO, N_EPISODES)


def select_action(state, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(nA)

    values = Q[state]
    return int(np.random.choice(np.flatnonzero(values == values.max())))


for e in range(N_EPISODES):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            renderer.close()
            env.close()
            raise SystemExit

    traces = np.zeros_like(Q)
    state, _ = env.reset()
    terminated, truncated = False, False
    action = select_action(state, epsilons[e])

    while not (terminated or truncated):
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_action = select_action(next_state, epsilons[e])

        td_error = reward + GAMMA * Q[next_state, next_action] * (not done) - Q[state, action]
        traces[state, action] += 1
        Q += alphas[e] * td_error * traces
        traces *= GAMMA * LAMBDA

        state, action = next_state, next_action

        renderer.set_values(np.max(Q, axis=1))
        renderer.render(state)
        pygame.display.set_caption(
            f"FrozenLake SARSA(lambda={LAMBDA}) - ep {e + 1}/{N_EPISODES}"
        )
        pygame.time.wait(int(STEP_DELAY * 1000))

    pygame.time.wait(int(EPISODE_DELAY * 1000))

pygame.display.set_caption("FrozenLake SARSA(lambda) (evaluation)")

obs, info = env.reset()
renderer.set_values(np.max(Q, axis=1))
renderer.render(obs)

terminated, truncated = False, False
while not (terminated or truncated):
    pygame.time.wait(300)
    action = select_action(obs, epsilon=0.0)
    obs, reward, terminated, truncated, info = env.step(action)
    renderer.render(obs)

renderer.close()
env.close()
