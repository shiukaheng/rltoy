import gymnasium as gym

env = gym.make(
    "FrozenLake-v1",
    map_name="4x4",
    is_slippery=False,
    render_mode=None
)

# Env: 
# .reset() -> (observation, Dict)
# .step(action) -> (observation, reward, terminated bool, truncated bool, info)
# .render()
# .close()

# Space:
# For describing sets: observations / actions

# The observation is one integer: 0, 1, ..., 15.
print(env.observation_space)       # Discrete(16)
print(env.observation_space.n)     # 16 possible states

# The action is one integer: 0, 1, 2, or 3.
print(env.action_space)            # Discrete(4)
print(env.action_space.n)          # 4 possible actions

# Ask the action space to produce a valid random action.
print(env.action_space.sample())

env.close()