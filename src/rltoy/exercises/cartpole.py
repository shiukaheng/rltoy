import gymnasium as gym
import torch
import torch.nn as nn

env = gym.make("CartPole-v1", render_mode="human")
state, info = env.reset()

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

policy = CartPolePolicy()
policy.train()

try:
    while True:
        state_tensor = torch.from_numpy(state).float()
        probs = policy(state_tensor)
        action = torch.multinomial(probs, 1).item()
        state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            state, info = env.reset()
except KeyboardInterrupt:
    env.close()
