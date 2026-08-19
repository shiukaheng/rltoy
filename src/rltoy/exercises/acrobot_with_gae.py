import gymnasium as gym
import math
import torch
import torch.nn as nn
import torch.functional as F
from tqdm import trange

DISCOUNT_FACTOR = 0.99  # how much future rewards are worth relative to immediate ones
GOAL_HEIGHT = 1.0
TIP_DISTANCE_PENALTY = 0.1  # reward shaping: penalize being far below the goal line
SPIN_PENALTY = 0.05  # reward shaping: penalize second arm spinning more than one full cycle


def tip_height(state):
    cos1, sin1, cos2, sin2 = state[0], state[1], state[2], state[3]
    # height of the tip relative to the top pivot, using the two-link kinematics
    return -cos1 - (cos1 * cos2 - sin1 * sin2)


def encode_state(state):
    return torch.from_numpy(state).float()  # gymnasium state -> float tensor, shape [state_dim=6]


class AcrobotPolicy(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.layers = nn.Linear(6, n_actions)  # 6-dimensional state -> action logits

    def forward(self, state):
        # state: [6] or [batch, 6]; logits and probabilities: [3] or [batch, 3]
        logits = self.layers(state)
        return torch.softmax(logits, dim=-1)  # normalize over the final action dimension

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
        # state: [6] or [batch, 6]; pred_value: [1] or [batch, 1]
        pred_value = self.layers(state)
        return pred_value

def run_episode(env, pi, policy_opt, v, value_opt):
    # Collect and update from a single episode.

    state_n, info = env.reset() # reset for a new episode

    max_tip_height = -float('inf')

    # iterate until environment signals termination
    while True:

        state_n_tensor = encode_state(state_n) # shape [6]

        # sample policy (no gradient)
        with torch.no_grad():
            probs = pi(state_n_tensor)  # shape [3]
        action = torch.multinomial(probs, 1).item()  # sample an action from the policy distribution

        # actually run the action and see the results
        state_np1, reward, terminated, truncated, info = env.step(action)
        state_np1_tensor = encode_state(state_np1)

        # shaping reward: penalize distance below the goal height, based on the best
        # height achieved so far in this episode (not the current height)
        current_height = tip_height(state_np1)
        max_tip_height = max(max_tip_height, current_height)
        reward -= TIP_DISTANCE_PENALTY * max(0.0, GOAL_HEIGHT - max_tip_height)

        # calculate the bootstrapped reward
        # terminal states have no future value; time-limit truncations still bootstrap.
        with torch.no_grad():
            if terminated:
                bootstrapped_reward = torch.tensor(reward, dtype=state_n_tensor.dtype)
            else:
                bootstrapped_reward = reward + DISCOUNT_FACTOR * v(state_np1_tensor)

        # trained our bootstrapped reward predictor (for baseline / advantage estimation)
        pred_bootstrapped_reward = v(state_n_tensor)

        # calculate loss for our policy
        advantage = bootstrapped_reward - torch.detach(v(state_n_tensor))
        action_log_prob = torch.log(pi(state_n_tensor)[action])
        policy_loss = -advantage * action_log_prob
        policy_opt.zero_grad()
        policy_loss.backward()
        policy_opt.step()

        # calculate loss for our value estimator network
        value_loss = (bootstrapped_reward - pred_bootstrapped_reward) ** 2
        value_opt.zero_grad()
        value_loss.backward()
        value_opt.step()

        state_n = state_np1

        # terminate if done
        if terminated or truncated:
            break

# instantiate policy and optimizer
pi = AcrobotPolicy(3)  # 3 actions: left, right, no-torque
pi.train()
policy_opt = torch.optim.Adam(pi.parameters(), lr=0.02)

v = AcrobotValueEstimator()
v.train()
value_opt = torch.optim.Adam(v.parameters(), lr=1e-3)

episode = 0
env = None

try:
    while True:
        # train for 100 episodes without rendering (faster)
        env = gym.make("Acrobot-v1")
        for _ in trange(100, desc="Training", leave=False):
            run_episode(env, pi, policy_opt, v, value_opt)
            episode += 1
        print(f"Completed episode {episode}")
        env.close()

        # run one demo episode with rendering to visualize progress
        env = gym.make("Acrobot-v1", render_mode="human")
        run_episode(env, pi, policy_opt, v, value_opt)
        print(f"Demo after episode {episode}")
        env.close()
except KeyboardInterrupt:
    if env is not None:
        env.close()
