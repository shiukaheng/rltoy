import gymnasium as gym
import torch
import torch.nn as nn

DISCOUNT_FACTOR = 0.99
GAE_LAMBDA = 0.95

GOAL_HEIGHT = 1.0
TIP_DISTANCE_PENALTY = 0.1
SPIN_PENALTY = 0.05


def tip_height(state):
    cos1, sin1, cos2, sin2 = state[0], state[1], state[2], state[3]
    return -cos1 - (cos1 * cos2 - sin1 * sin2)


def encode_state(state):
    return torch.from_numpy(state).float()


class AcrobotPolicy(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.layers = nn.Linear(6, n_actions)

    def forward(self, state):
        logits = self.layers(state)
        return torch.softmax(logits, dim=-1)


class AcrobotValueEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, state):
        return self.layers(state)


def run_episode(env, pi, policy_opt, v, value_opt, train):
    state, info = env.reset()

    visited_states = []
    selected_actions = []
    observed_rewards = []

    max_tip_height = -float("inf")

    while True:
        state_tensor = encode_state(state)

        # Sample action without building a gradient graph.
        with torch.no_grad():
            probs = pi(state_tensor)

        action = torch.multinomial(probs, 1).item()

        next_state, reward, terminated, truncated, info = env.step(action)

        # Reward shaping.
        current_height = tip_height(next_state)
        max_tip_height = max(max_tip_height, current_height)

        reward -= TIP_DISTANCE_PENALTY * max(
            0.0,
            GOAL_HEIGHT - max_tip_height,
        )

        # Store transition information.
        visited_states.append(state_tensor)
        selected_actions.append(action)
        observed_rewards.append(reward)

        state = next_state

        if terminated or truncated:
            break

    episode_return = sum(observed_rewards)

    if train:
        # ------------------------------------------------------------
        # Convert trajectory to tensors
        # ------------------------------------------------------------

        states = torch.stack(visited_states)                  # [T, 6]
        actions = torch.tensor(selected_actions)             # [T]
        rewards = torch.tensor(observed_rewards).float()     # [T]

        T = len(rewards)

        # ------------------------------------------------------------
        # Compute GAE
        #
        # δ_t = r_t + γ V(s_{t+1}) - V(s_t)
        #
        # A_t^GAE = δ_t + γ λ A_{t+1}^GAE
        # ------------------------------------------------------------

        with torch.no_grad():
            values = v(states).squeeze(1)  # [T]

            # The variable `state` is now the state after the final
            # transition, i.e. s_T.
            #
            # A genuinely terminated episode has V(s_T) = 0.
            # A time-limit truncation still bootstraps from V(s_T).
            if terminated:
                final_next_value = torch.tensor(0.0)
            else:
                final_next_value = v(encode_state(state)).squeeze()

            advantages = torch.zeros_like(rewards)

            gae = torch.tensor(0.0)

            # Work backwards through the trajectory.
            for t in reversed(range(T)):

                # For all non-final steps:
                #
                #     s_{t+1} = states[t + 1]
                #
                # so its value is values[t + 1].
                #
                # For the final step, use the separately evaluated s_T.
                if t == T - 1:
                    next_value = final_next_value
                else:
                    next_value = values[t + 1]

                delta = (
                    rewards[t]
                    + DISCOUNT_FACTOR * next_value
                    - values[t]
                )

                gae = (
                    delta
                    + DISCOUNT_FACTOR
                    * GAE_LAMBDA
                    * gae
                )

                advantages[t] = gae

            # Since
            #
            #     advantage ≈ target - V(s_t)
            #
            # we can reconstruct a target for the critic:
            #
            #     target = V(s_t) + advantage
            #
            value_targets = values + advantages

        # ------------------------------------------------------------
        # Policy update
        # ------------------------------------------------------------

        # Advantage normalization is optional but commonly useful.
        advantages_for_policy = (
            advantages - advantages.mean()
        ) / (
            advantages.std() + 1e-8
        )

        action_probs = pi(states)  # [T, 3]

        selected_action_probs = action_probs.gather(
            1,
            actions.unsqueeze(1),
        ).squeeze(1)

        log_probs = torch.log(selected_action_probs)

        policy_loss = -(
            log_probs * advantages_for_policy
        ).mean()

        policy_opt.zero_grad()
        policy_loss.backward()
        policy_opt.step()

        # ------------------------------------------------------------
        # Value function update
        # ------------------------------------------------------------

        pred_values = v(states).squeeze(1)

        value_loss = torch.mean(
            (pred_values - value_targets) ** 2
        )

        value_opt.zero_grad()
        value_loss.backward()
        value_opt.step()

    return episode_return


# ------------------------------------------------------------
# Instantiate policy and value function
# ------------------------------------------------------------

pi = AcrobotPolicy(3)
pi.train()

policy_opt = torch.optim.Adam(
    pi.parameters(),
    lr=0.02,
)

v = AcrobotValueEstimator()
v.train()

value_opt = torch.optim.Adam(
    v.parameters(),
    lr=1e-3,
)


# ------------------------------------------------------------
# Training loop
# ------------------------------------------------------------

episode = 0
env = None

try:
    while True:
        env = gym.make("Acrobot-v1")

        for _ in range(100):
            episode_return = run_episode(
                env,
                pi,
                policy_opt,
                v,
                value_opt,
                train=True,
            )

            episode += 1

        print(
            f"Episode {episode} return: "
            f"{episode_return:.0f}"
        )

        env.close()

        # Demo episode.
        env = gym.make(
            "Acrobot-v1",
            render_mode="human",
        )

        demo_return = run_episode(
            env,
            pi,
            policy_opt,
            v,
            value_opt,
            train=False,
        )

        print(
            f"Demo after episode {episode} "
            f"return: {demo_return:.0f}"
        )

        env.close()

except KeyboardInterrupt:
    if env is not None:
        env.close()