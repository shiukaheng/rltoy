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
    visited_states = []
    selected_actions = []
    observed_rewards = []
    while True:
        state_tensor = encode_state(state)
        with torch.no_grad():
            probs = policy(state_tensor)
        action = torch.multinomial(probs, 1).item()
        state, reward, terminated, truncated, info = env.step(action)
        visited_states.append(state_tensor)
        selected_actions.append(action)
        observed_rewards.append(reward)
        if terminated or truncated:
            break

    episode_return = sum(observed_rewards)

    if train:
        states = torch.stack(visited_states)
        actions = torch.tensor(selected_actions)
        rewards = torch.tensor(observed_rewards)
        action_probs = policy(states)

        discounts = DISCOUNT_FACTOR ** torch.arange(rewards.shape[0])
        discounted_rewards = rewards * discounts
        returns = torch.flip(
            torch.cumsum(torch.flip(discounted_rewards, dims=(0,)), dim=0),
            dims=(0,),
        ) / discounts

        selected_action_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        episode_loss = -(torch.log(selected_action_probs) * returns).sum()
        optimizer.zero_grad()
        episode_loss.backward()
        optimizer.step()

    return episode_return


policy = CartPolePolicy()
policy.train()
optimizer = torch.optim.Adam(policy.parameters(), lr=0.02)

episode = 0
env = None

try:
    while True:
        env = gym.make("CartPole-v1")
        for _ in range(100):
            episode_return = run_episode(env, policy, optimizer, train=True)
            episode += 1
        print(f"Episode {episode} return: {episode_return:.0f}")
        env.close()

        env = gym.make("CartPole-v1", render_mode="human")
        demo_return = run_episode(env, policy, optimizer, train=False)
        print(f"Demo after episode {episode} return: {demo_return:.0f}")
        env.close()
except KeyboardInterrupt:
    if env is not None:
        env.close()
