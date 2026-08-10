"""Validate GraphWorld environment and planner integration.

Run with: python3 tests/test_graph_world.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph_world import GraphWorldEnv, _compile_P, _validate_graph_spec, load_graph_spec, make_graph_world
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

# ── risk_vs_delay graph ─────────────────────────────────

spec = load_graph_spec("graphs/risk_vs_delay.json")
check(spec["name"] == "risk_vs_delay", "risk_vs_delay loads")

env2, V2 = make_graph_world("graphs/risk_vs_delay.json", gamma=0.95)
check(len(V2) == 4, f"4 states, got {len(V2)}")
check(V2[2] == 0.0 and V2[3] == 0.0, "terminal states have V=0")
check(env2.terminal_states()[2], "state 2 (goal) is terminal")
check(env2.terminal_states()[3], "state 3 (loss) is terminal")
check(not env2.terminal_states()[0], "state 0 (start) is not terminal")
env2.close()

# optimal policy at gamma=0.95: safe (Q_safe = 9.5 > Q_risky = 7.6)
env3, V3 = make_graph_world("graphs/risk_vs_delay.json", gamma=0.95)
P3 = env3.P
Q = np.zeros((env3.n_states, env3.n_actions))
for s in range(env3.n_states):
    for a in range(env3.n_actions):
        for prob, ns, r, done in P3[s][a]:
            Q[s, a] += prob * (r + (0.0 if done else 0.95 * V3[ns]))
optimal_start = np.argmax(Q[0])
check(optimal_start == 0, f"optimal at start is safe(0), got {optimal_start}")
env3.close()

# offline policy at gamma=0.70: risky (Q_safe = 7.0 < Q_risky = 7.6)
env4, V4 = make_graph_world("graphs/risk_vs_delay.json", gamma=0.70)
Q = np.zeros((env4.n_states, env4.n_actions))
for s in range(env4.n_states):
    for a in range(env4.n_actions):
        for prob, ns, r, done in env4.P[s][a]:
            Q[s, a] += prob * (r + (0.0 if done else 0.70 * V4[ns]))
optimal_start = np.argmax(Q[0])
check(optimal_start == 1, f"optimal at start is risky(1), got {optimal_start}")
env4.close()

# ── summary ─────────────────────────────────────────────

print(f"\n{passed} passed, {failed} failed")
if failed:
    sys.exit(1)