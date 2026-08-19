import gymnasium as gym
import math
import torch
import torch.nn as nn
import torch.functional as F

DISCOUNT_FACTOR = 0.99  # how much future rewards are worth relative to immediate ones
GOAL_HEIGHT = 1.0
TIP_DISTANCE_PENALTY = 0.1  # reward shaping: penalize being far below the goal line
SPIN_PENALTY = 0.05  # reward shaping: penalize second arm spinning more than one full cycle


def tip_height(state):
    cos1, sin1, cos2, sin2 = state[0], state[1], state[2], state[3]
    # height of the tip relative to the top pivot, using the two-link kinematics
    return -cos1 - (cos1 * cos2 - sin1 * sin2)


def encode_state(state):
    return torch.from_numpy(state).float()  # gymnasium state is a numpy array; convert to tensor


class AcrobotPolicy(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.layers = nn.Linear(6, n_actions)  # 6-dimensional state -> action logits

    def forward(self, state):
        logits = self.layers(state)
        return torch.softmax(logits, dim=-1)  # outputs a probability distribution over actions

class AcrobotValueEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(6,32),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,1)
        )

    def forward(self, state):
        pred_value = self.layers(state)
        return pred_value

def run_episode(env, pi, policy_opt, v, value_opt, train):
    # collect a single episode; if train=True, run a REINFORCE update afterwards

    state, info = env.reset() # reset for a new episode

    # create lists for trajectory
    visited_states = []
    selected_actions = []
    observed_rewards = []

    max_tip_height = -float('inf')
    prev_theta2 = None
    cumulative_spin = 0.0

    # iterate until environment signals termination
    while True:

        state_tensor = encode_state(state) # turn gymnasium state into tensor

        # sample policy (no gradient)
        with torch.no_grad():
            probs = pi(state_tensor)
        action = torch.multinomial(probs, 1).item()  # sample an action from the policy distribution

        # actually run the action and see the results
        state, reward, terminated, truncated, info = env.step(action)

        # shaping reward: penalize distance below the goal height, based on the best
        # height achieved so far in this episode (not the current height)
        current_height = tip_height(state)
        max_tip_height = max(max_tip_height, current_height)
        reward -= TIP_DISTANCE_PENALTY * max(0.0, GOAL_HEIGHT - max_tip_height)

        # penalty for second arm spinning more than one full cycle
        theta2 = math.atan2(state[3], state[2])
        if prev_theta2 is not None:
            dtheta = (theta2 - prev_theta2 + math.pi) % (2 * math.pi) - math.pi
            cumulative_spin += abs(dtheta)
        prev_theta2 = theta2
        reward -= SPIN_PENALTY * max(0.0, cumulative_spin - 2 * math.pi)

        # add to trajectory
        visited_states.append(state_tensor)
        selected_actions.append(action)
        observed_rewards.append(reward)

        # terminate if done
        if terminated or truncated:
            break

    
    episode_return = sum(observed_rewards)

    if train:
        # --- REINFORCE update ---

        # vectorizing the data required for training: states, actions, rewards
        states = torch.stack(visited_states)
        actions = torch.tensor(selected_actions)
        rewards = torch.tensor(observed_rewards)
        action_probs = pi(states)

        # --- compute discounted returns G_t for each timestep ---
        # for each t:  G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ... + γ^{T-1-t}·r_{T-1}
        #
        # vectorized trick:  multiply every r_t by γ^t, then a reverse cumsum
        # sums from the end, and finally divide by γ^t to undo the premultiplication.
        #
        #   discounted_rewards = [γ⁰·r₀,  γ¹·r₁,  γ²·r₂,  ...,  γ^{T-1}·r_{T-1}]
        #   reverse → cumsum → reverse → divide elementwise by discounts
        #   → [G₀, G₁, G₂, ..., G_{T-1}]
        discounts = DISCOUNT_FACTOR ** torch.arange(rewards.shape[0], dtype=torch.float32)
        discounted_rewards = rewards * discounts
        returns = torch.flip(
            torch.cumsum(torch.flip(discounted_rewards, dims=(0,)), dim=0),
            dims=(0,),
        ) / discounts

        # --- REINFORCE loss:  -Σ_t log π(a_t | s_t) · G_t  ---
        # for each timestep t:  loss_t = -log(π(a_t|s_t)) · G_t
        # total episode loss = sum of loss_t over all t
        selected_action_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        episode_loss = -(torch.log(selected_action_probs) * returns).sum()
        policy_opt.zero_grad()
        episode_loss.backward()
        policy_opt.step()

    return episode_return # returning this is just useful for visualization. not used for training.


# instantiate policy and optimizer
pi = AcrobotPolicy(3)  # 3 actions: left, right, no-torque
pi.train()
policy_opt = torch.optim.Adam(pi.parameters(), lr=0.02)

v = AcrobotValueEstimator()
v.train()
value_opt = torch.optim.Adam(pi.parameters(), lr=0.02)

episode = 0
env = None

try:
    while True:
        # train for 100 episodes without rendering (faster)
        env = gym.make("Acrobot-v1")
        for _ in range(100):
            episode_return = run_episode(env, pi, policy_opt, v, value_opt, train=True)
            episode += 1
        print(f"Episode {episode} return: {episode_return:.0f}")
        env.close()

        # run one demo episode with rendering to visualize progress
        env = gym.make("Acrobot-v1", render_mode="human")
        demo_return = run_episode(env, pi, policy_opt, v, value_opt, train=False)
        print(f"Demo after episode {episode} return: {demo_return:.0f}")
        env.close()
except KeyboardInterrupt:
    if env is not None:
        env.close()
