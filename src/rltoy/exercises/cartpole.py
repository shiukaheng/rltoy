import gymnasium as gym
import torch.nn as nn

env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset()

class CartPolePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4,4),
            nn.ReLU(),
            nn.Linear(4,4),
            nn.ReLU(),
            nn.Linear(4,2),
            nn.Softmax(dim=-1)
        )
    def forward(self, state):
        """
        (4,): (pos, vel, ang, ang_vel) ->
        (2,): (left_logit, right_logit)
        """
        return self.layers(state)


for _ in range(1_000):
    action = env.action_space.sample()  # Random policy.
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()

env.close()
