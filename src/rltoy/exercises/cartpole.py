import gymnasium as gym
import torch
import torch.nn as nn

DISCOUNT_FACTOR = 0.99


def encode_state(state):
    state = torch.from_numpy(state).float()
    return torch.stack((state[0], state[1], torch.cos(state[2]), torch.sin(state[2]), state[3]))


class CartPolePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mirror_signs", torch.tensor([-1.0, -1.0, 1.0, -1.0, -1.0]))
        self.layers = nn.Linear(5, 1)

    def forward(self, state):
        preference = self.layers(state).squeeze(-1)
        mirrored_preference = self.layers(state * self.mirror_signs).squeeze(-1)
        preference = (preference - mirrored_preference) / 2
        logits = torch.stack((-preference, preference), dim=-1)
        return torch.softmax(logits, dim=-1)


def run_episode(env, policy, optimizer, train):
    state, info = env.reset()
    trajectory_buffer = []
    while True:
        state_tensor = encode_state(state)
        probs = policy(state_tensor)
        action = torch.multinomial(probs, 1).item()
        state, reward, terminated, truncated, info = env.step(action)
        trajectory_buffer.append((state_tensor, probs, action, torch.tensor([reward])))
        if terminated or truncated:
            trajectory_buffer.append((encode_state(state), None, None, None))
            break

    episode_return = sum(item[3].item() if item[3] is not None else 0 for item in trajectory_buffer[:-1])

    if train:
        trajectory_buffer = list(zip(*trajectory_buffer))
        trajectory_buffer = [[x for x in l if x is not None] for l in trajectory_buffer]
        states = torch.stack(trajectory_buffer[0])
        action_probs = torch.stack(trajectory_buffer[1])
        actions = torch.tensor(trajectory_buffer[2])
        rewards = torch.stack(trajectory_buffer[3])

        optimizer.zero_grad()
        episode_loss = torch.tensor(0.0)
        for t in range(int(states.shape[0]) - 1):
            reward_array = rewards[t:]
            coeffs = torch.full((reward_array.shape[0],), DISCOUNT_FACTOR) ** torch.arange(reward_array.shape[0])
            return_ = torch.dot(reward_array.squeeze(-1), coeffs)
            episode_loss = episode_loss + (-torch.log(action_probs[t, actions[t]]) * return_)
        episode_loss.backward()
        optimizer.step()

    return episode_return


policy = CartPolePolicy()
policy.train()
optimizer = torch.optim.Adam(policy.parameters(), lr=0.02)

headless_budget = 100
headless_remaining = 0
episode = 0

try:
    while True:
        if headless_remaining <= 0:
            headless_remaining = headless_budget

        train = headless_remaining > 1
        render_mode = None if train else "human"
        env = gym.make("CartPole-v1", render_mode=render_mode)

        episode_return = run_episode(env, policy, optimizer, train)
        env.close()

        episode += 1
        if train:
            headless_remaining -= 1
            if episode % 100 == 0:
                print(f"Episode {episode} return: {episode_return:.0f}")
        else:
            headless_remaining = 0
            print(f"Demo episode {episode} return: {episode_return:.0f}")
except KeyboardInterrupt:
    pass