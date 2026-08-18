import gymnasium as gym
import torch
import torch.nn as nn
import time

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
        trajectory_buffer = []
        while True:
            state_tensor = torch.from_numpy(state).float()
            probs = policy(state_tensor)
            action = torch.multinomial(probs, 1).item()
            state, reward, terminated, truncated, info = env.step(action)
            trajectory_buffer.append((state_tensor, probs, torch.tensor([reward]))) # sn, an, rnp1
            if terminated or truncated:
                trajectory_buffer.append((torch.from_numpy(state).float(), None, None)) # last state without action and reward
                break
        trajectory_buffer = list(zip(*trajectory_buffer)) # snp1
        trajectory_buffer = [[x for x in l if x is not None] for l in trajectory_buffer]
        states, action_probs, rewards = [torch.stack(x) for x in trajectory_buffer]
        print(states.shape, action_probs.shape, rewards.shape)
        state, info = env.reset()
except KeyboardInterrupt:
    env.close()
