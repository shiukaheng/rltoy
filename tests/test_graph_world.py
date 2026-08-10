"""Validate GraphWorld environment and planner integration.

Run with: python3 tests/test_graph_world.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph_world import GraphWorldEnv, _validate_graph_spec, make_graph_world
from bettermdptools.algorithms.planner import Planner

SPEC = {
    "name": "test",
    "start_state": "start",
    "actions": {"safe": {}, "risky": {}},
    "states": {
        "start": {
            "label": "Start",
            "position": [0.1, 0.5],
            "actions": {
                "safe": [{"probability": 1.0, "next_state": "goal", "reward": 10.0, "terminated": True}],
                "risky": [
                    {"probability": 0.6, "next_state": "goal", "reward": 20.0, "terminated": True},
                    {"probability": 0.4, "next_state": "loss", "reward": -15.0, "terminated": True},
                ],
            },
        },
        "goal": {"label": "Goal", "position": [0.9, 0.3], "terminal": True},
        "loss": {"label": "Loss", "position": [0.9, 0.7], "terminal": True},
    },
}

passed = 0
failed = 0


def check(cond, msg):
    global passed, failed
    if cond:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL: {msg}")


# ── validation ──────────────────────────────────────────

_validate_graph_spec(SPEC)
check(True, "valid spec passes validation")

try:
    bad = json.loads(json.dumps(SPEC))
    del bad["states"]["start"]["actions"]["safe"]
    _validate_graph_spec(bad)
    check(False, "missing action should fail")
except ValueError:
    check(True, "missing action fails validation")

try:
    bad = json.loads(json.dumps(SPEC))
    bad["states"]["start"]["actions"]["risky"][0]["probability"] = 0.3
    _validate_graph_spec(bad)
    check(False, "non-unit probability sum should fail")
except ValueError:
    check(True, "non-unit probability sum fails validation")

try:
    bad = json.loads(json.dumps(SPEC))
    bad["states"]["start"]["actions"]["safe"][0]["next_state"] = "nope"
    _validate_graph_spec(bad)
    check(False, "unknown next_state should fail")
except ValueError:
    check(True, "unknown next_state fails validation")

try:
    bad = json.loads(json.dumps(SPEC))
    bad["start_state"] = "nope"
    _validate_graph_spec(bad)
    check(False, "unknown start_state should fail")
except ValueError:
    check(True, "unknown start_state fails validation")

try:
    bad = json.loads(json.dumps(SPEC))
    bad["states"]["goal"]["actions"] = {"safe": []}
    _validate_graph_spec(bad)
    check(False, "terminal state with actions should fail")
except ValueError:
    check(True, "terminal state with actions fails validation")

# ── compilation ─────────────────────────────────────────

env = GraphWorldEnv(SPEC)
check(env.n_states == 3, f"n_states == 3, got {env.n_states}")
check(env.n_actions == 2, f"n_actions == 2, got {env.n_actions}")
check(isinstance(env.P, dict), "P is a dict")
check(len(env.P) == 3, f"P has 3 states, got {len(env.P)}")

# terminal states should have absorbing self-transitions for both actions
for a in range(2):
    for s in [1, 2]:  # goal, loss
        transitions = env.P[s][a]
        check(len(transitions) == 1, f"terminal state {s} action {a} has 1 transition")
        check(transitions[0] == (1.0, s, 0.0, True), f"terminal state {s} is absorbing")

# start state transitions
safe_trans = env.P[0][0]
check(len(safe_trans) == 1, "start-safe has 1 outcome")
check(safe_trans[0] == (1.0, 1, 10.0, True), f"start-safe: {safe_trans[0]}")

risky_trans = env.P[0][1]
check(len(risky_trans) == 2, "start-risky has 2 outcomes")
total_p = sum(t[0] for t in risky_trans)
check(abs(total_p - 1.0) < 1e-9, f"start-risky probs sum to 1.0 ({total_p})")

# ── seeding ─────────────────────────────────────────────

obs1, _ = env.reset(seed=42)
obs2, _ = env.reset(seed=42)
check(obs1 == obs2, "same seed gives same reset")

o1, r1, t1, _, _ = env.step(1)
env.reset(seed=42)
o2, r2, t2, _, _ = env.step(1)
check(o1 == o2 and r1 == r2, "same seed gives same step result")

# ── planner ─────────────────────────────────────────────

V, _, info = Planner(env.P).value_iteration(gamma=0.95)
V = np.array(V)
q_safe = 10.0
q_risky = 0.6 * 20 + 0.4 * (-15)
expected_v0 = max(q_safe, q_risky)
check(abs(V[0] - expected_v0) < 0.01, f"planner V[start]={V[0]:.3f} ≈ expected {expected_v0:.3f}")
check(V[1] == 0.0, f"planner V[goal] == 0.0, got {V[1]}")
check(V[2] == 0.0, f"planner V[loss] == 0.0, got {V[2]}")

env.close()

# ── branching_risk graph ─────────────────────────────────

env, values = make_graph_world("graphs/branching_risk.json", gamma=0.95)
check(env.n_states == 14, f"branching_risk has 14 states, got {env.n_states}")
check(env.n_actions == 3, f"branching_risk has 3 actions, got {env.n_actions}")
check(abs(values[env.state_index("start")] - 42.86875) < 0.01, "branching_risk start value")

policy = []
for state in range(env.n_states):
    action_values = [
        sum(
            probability * (reward + (0.0 if done else 0.95 * values[next_state]))
            for probability, next_state, reward, done in env.P[state][action]
        )
        for action in range(env.n_actions)
    ]
    policy.append(int(np.argmax(action_values)))

check(policy[env.state_index("start")] == env.action_index("c"), "start selects branch 3")
for state_name in ("b2_l1", "b2_l2", "b3_l1", "b3_l2", "b3_l3"):
    check(policy[env.state_index(state_name)] == env.action_index("a"), f"{state_name} advances with A")
env.close()

# ── summary ─────────────────────────────────────────────

print(f"\n{passed} passed, {failed} failed")
if failed:
    sys.exit(1)
