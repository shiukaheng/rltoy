"""Generate a GraphWorld JSON spec with force-directed layout.

Usage: python3 scripts/layout_graph.py > graphs/branching_risk.json
"""

import json
import math
import sys


def layered_force_layout(nodes, edges, levels, iterations=250):
    """Refine a tree layout with horizontal spring and repulsion forces."""
    by_level = {}
    for name, level in levels.items():
        by_level.setdefault(level, []).append(name)

    children = {name: [] for name in nodes}
    for src, dst in edges:
        children[src].append(dst)

    def ordered_leaves(name):
        if not children[name]:
            return [name]
        return [leaf for child in children[name] for leaf in ordered_leaves(child)]

    leaves = ordered_leaves("start")
    base_x = {}
    for i, name in enumerate(leaves):
        base_x[name] = (i + 1) / (len(leaves) + 1)

    def place_parent(name):
        if name in base_x:
            return base_x[name]
        base_x[name] = sum(place_parent(child) for child in children[name]) / len(children[name])
        return base_x[name]

    place_parent("start")
    pos = {}
    max_level = max(levels.values())
    for name in nodes:
        pos[name] = [base_x[name], 0.08 + 0.84 * levels[name] / max_level]

    for iteration in range(iterations):
        temperature = 0.06 * (1 - iteration / iterations)
        displacement = {name: 0.0 for name in nodes}

        for names in by_level.values():
            for i, left in enumerate(names):
                for right in names[i + 1 :]:
                    delta = pos[left][0] - pos[right][0]
                    distance = max(abs(delta), 0.02)
                    force = 0.0015 / (distance * distance)
                    displacement[left] += math.copysign(force, delta)
                    displacement[right] -= math.copysign(force, delta)

        for src, dst in edges:
            delta = pos[dst][0] - pos[src][0]
            force = 0.025 * delta
            displacement[src] += force
            displacement[dst] -= force

        for name in nodes:
            displacement[name] += 0.2 * (base_x[name] - pos[name][0])
            dx = max(-temperature, min(temperature, displacement[name]))
            pos[name][0] = max(0.05, min(0.95, pos[name][0] + dx))

    return {name: tuple(coords) for name, coords in pos.items()}


def build_spec():
    tree = {
        "start": ["b1_win", "b2_l1", "b3_l1"],
        "b2_l1": ["b2_l2", "b2_lose_l1"],
        "b2_l2": ["b2_win", "b2_lose_l2"],
        "b3_l1": ["b3_l2", "b3_lose_l1"],
        "b3_l2": ["b3_l3", "b3_lose_l2"],
        "b3_l3": ["b3_win", "b3_lose_l3"],
    }

    nodes = list(tree)
    for children in tree.values():
        for c in children:
            if c not in nodes:
                nodes.append(c)

    edges = [(src, dst) for src, children in tree.items() for dst in children]
    levels = {
        "start": 0,
        "b1_win": 1,
        "b2_l1": 1,
        "b3_l1": 1,
        "b2_l2": 2,
        "b2_lose_l1": 2,
        "b3_l2": 2,
        "b3_lose_l1": 2,
        "b2_win": 3,
        "b2_lose_l2": 3,
        "b3_l3": 3,
        "b3_lose_l2": 3,
        "b3_win": 4,
        "b3_lose_l3": 4,
    }
    positions = layered_force_layout(nodes, edges, levels)

    spec = {
        "name": "branching_risk",
        "start_state": "start",
        "actions": {
            "a": {"label": "A", "key": "a"},
            "b": {"label": "B", "key": "b"},
            "c": {"label": "C", "key": "c"},
        },
        "states": {},
    }

    labels = {
        "start": "Start",
        "b1_win": "B1: +5",
        "b2_l1": "B2 L1",
        "b2_l2": "B2 L2",
        "b2_win": "B2: +20",
        "b2_lose_l1": "B2 L1: -10",
        "b2_lose_l2": "B2 L2: -10",
        "b3_l1": "B3 L1",
        "b3_l2": "B3 L2",
        "b3_l3": "B3 L3",
        "b3_win": "B3: +50",
        "b3_lose_l1": "B3 L1: -15",
        "b3_lose_l2": "B3 L2: -15",
        "b3_lose_l3": "B3 L3: -15",
    }

    for name in nodes:
        terminal = "win" in name or "lose" in name
        x, y = positions[name]
        state_def = {
            "label": labels.get(name, name),
            "position": [round(x, 4), round(y, 4)],
        }
        if terminal:
            state_def["terminal"] = True
        spec["states"][name] = state_def

    transitions = {
        "start": {
            "a": [("b1_win", 5.0, True)],
            "b": [("b2_l1", 0.0, False)],
            "c": [("b3_l1", 0.0, False)],
        },
        "b2_l1": {
            "a": [("b2_l2", 0.0, False)],
            "b": [("b2_lose_l1", -10.0, True)],
            "c": [("b2_lose_l1", -10.0, True)],
        },
        "b2_l2": {
            "a": [("b2_win", 20.0, True)],
            "b": [("b2_lose_l2", -10.0, True)],
            "c": [("b2_lose_l2", -10.0, True)],
        },
        "b3_l1": {
            "a": [("b3_l2", 0.0, False)],
            "b": [("b3_lose_l1", -15.0, True)],
            "c": [("b3_lose_l1", -15.0, True)],
        },
        "b3_l2": {
            "a": [("b3_l3", 0.0, False)],
            "b": [("b3_lose_l2", -15.0, True)],
            "c": [("b3_lose_l2", -15.0, True)],
        },
        "b3_l3": {
            "a": [("b3_win", 50.0, True)],
            "b": [("b3_lose_l3", -15.0, True)],
            "c": [("b3_lose_l3", -15.0, True)],
        },
    }

    for state_name, actions in transitions.items():
        state_def = spec["states"][state_name]
        if "terminal" in state_def:
            continue
        state_def["actions"] = {}
        for a_name, outcomes in actions.items():
            state_def["actions"][a_name] = [
                {
                    "probability": 1.0,
                    "next_state": ns,
                    "reward": r,
                }
                | ({"terminated": True} if term else {})
                for ns, r, term in outcomes
            ]

    return spec


if __name__ == "__main__":
    spec = build_spec()
    json.dump(spec, sys.stdout, indent=2)
    sys.stdout.write("\n")
