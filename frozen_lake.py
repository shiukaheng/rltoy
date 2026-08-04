"""Shared FrozenLake environment factory."""

import copy

import gymnasium as gym
from bettermdptools.algorithms.planner import Planner


def _penalize_steps(P, cost):
    P = copy.deepcopy(P)
    for actions in P.values():
        for transitions in actions.values():
            for i, (prob, ns, r, done) in enumerate(transitions):
                if not done:
                    transitions[i] = (prob, ns, r + cost, done)
    return P


def make_frozen_lake(
    map_name="FrozenLake-v1",
    is_slippery=False,
    step_cost=None,
):
    env = gym.make(map_name, is_slippery=is_slippery)

    P = env.unwrapped.P
    if step_cost is not None:
        P = _penalize_steps(P, step_cost)

    V, _, _ = Planner(P).value_iteration()

    return env, V