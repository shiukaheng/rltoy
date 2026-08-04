import gymnasium as gym

env = gym.make(
    "FrozenLake-v1",
    render_mode="human",
    is_slippery=False,
)

obs, info = env.reset()

terminated = truncated = False

while not (terminated or truncated):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

env.close()