import gymnasium as gym

env = gym.make(
    "FrozenLake-v1",
    map_name="4x4",
    is_slippery=False,
    render_mode=None
)

o, info = env.reset()

while True:
    action = env.action_space.sample()
    print(action)
    o, r, term_b, trunc_b, info = env.step(action)
    if term_b or trunc_b:
        break

env.close()