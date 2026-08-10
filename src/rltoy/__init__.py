"""Pedagogical reinforcement-learning environments and algorithms."""

from gymnasium.envs.registration import register


register(
    id="RLToy/BranchingRisk-v0",
    entry_point="rltoy.envs.graph_world:make_branching_risk",
)
