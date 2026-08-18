import gymnasium as gym
import torch
import torch.nn as nn
import time

env = gym.make("CartPole-v1", render_mode="human")
state, info = env.reset()

DISCOUNT_FACTOR = 0.99

def encode_state(state):
    state = torch.from_numpy(state).float()
    return torch.stack((state[0], state[1], torch.cos(state[2]), torch.sin(state[2]), state[3]))

class CartPolePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(5,4),
            nn.ReLU(),
            nn.Linear(4,4),
            nn.ReLU(),
            nn.Linear(4,2),
            nn.Softmax(dim=-1)
        )
    def forward(self, state):
        """
        (5,): (pos, vel, cos_ang, sin_ang, ang_vel) ->
        (2,): (left_logit, right_logit)
        """
        return self.layers(state)

policy = CartPolePolicy()
policy.train()
optimizer = torch.optim.Adam(policy.parameters(), lr=0.02)

try:
    episode = 0
    while True:
        trajectory_buffer = []
        while True:
            state_tensor = encode_state(state)
            probs = policy(state_tensor)
            action = torch.multinomial(probs, 1).item()
            state, reward, terminated, truncated, info = env.step(action)
            trajectory_buffer.append((state_tensor, probs, action, torch.tensor([reward]))) # sn, an, rnp1
            if terminated or truncated:
                trajectory_buffer.append((encode_state(state), None, None, None)) # last state without action and reward
                break
        trajectory_buffer = list(zip(*trajectory_buffer)) # snp1
        trajectory_buffer = [[x for x in l if x is not None] for l in trajectory_buffer]
        states = torch.stack(trajectory_buffer[0])
        action_probs = torch.stack(trajectory_buffer[1])
        actions = torch.tensor(trajectory_buffer[2])
        rewards = torch.stack(trajectory_buffer[3])
        print(states.shape, action_probs.shape, rewards.shape)
        optimizer.zero_grad()
        episode_loss = torch.tensor(0.0)
        for t in range(int(states.shape[0])-1):
            reward_array = rewards[t:]
            coeffs_pow = torch.arange(reward_array.shape[0])
            coeffs_base = torch.full(coeffs_pow.shape, fill_value=DISCOUNT_FACTOR)
            coeffs = torch.pow(coeffs_base, coeffs_pow)
            return_ = torch.dot(reward_array.squeeze(-1), coeffs)
            episode_loss = episode_loss + (-torch.log(action_probs[t, actions[t]]) * return_)
        episode_loss.backward()
        optimizer.step()
        episode += 1
        print(f"Episode {episode} return: {rewards.sum().item():.0f}")
        state, info = env.reset()
except KeyboardInterrupt:
    env.close()
