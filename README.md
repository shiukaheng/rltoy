# rltoy

`rltoy` is a small laboratory for learning reinforcement learning by reading and changing working implementations.

Its design rule is simple: an algorithm's training loop and update rule live together in one readable file. Shared code is only for mechanics that are not the algorithm itself, such as epsilon-greedy action selection, schedules, and result recording.

## Run

Install the project with `uv sync`, then run a learner headlessly:

```bash
uv run rltoy-q-learning --episodes 1000
uv run rltoy-sarsa --episodes 1000
uv run rltoy-sarsa-lambda --episodes 1000 --trace-decay 0.8
```

Animate a learner on the bundled graph:

```bash
uv run rltoy-q-learning --render --render-delay-ms 50
uv run rltoy-interactive-graph
```

Compare the actual learner implementations with the graph's exact value-iteration solution:

```bash
uv run rltoy-compare-tabular --episodes 500 --runs 20
```

## Environments

`GraphWorldEnv` is a normal Gymnasium environment with discrete observations and actions:

```python
import gymnasium as gym
import rltoy

env = gym.make("RLToy/BranchingRisk-v0")
```

Graph JSON is an authoring format, not a new environment API. Load an authored graph directly with:

```python
from rltoy.envs import GraphWorldEnv

env = GraphWorldEnv.from_json("my_graph.json")
```

The required JSON fields are `start_state`, `actions`, `states`, action outcomes (`probability`, `next_state`, optional `reward`), and terminal states. Labels, positions, key bindings, and display colors are optional metadata used only by `GraphRenderer`.

Any finite discrete Gymnasium environment can be supplied to the tabular `train` functions. Wrap continuing environments with Gymnasium's `TimeLimit` rather than introducing a project-specific step-limit abstraction.

## Layout

```text
src/rltoy/algorithms/q_learning.py      # full Q-learning loop
src/rltoy/algorithms/sarsa.py           # full SARSA loop
src/rltoy/algorithms/sarsa_lambda.py    # full accumulating-trace SARSA(lambda) loop
src/rltoy/envs/graph_world.py           # JSON graph MDP and Gymnasium interface
src/rltoy/visualization/graph_renderer.py
```

Pygame visualization observes training snapshots but never enters learner logic. The first neural learner should follow the same boundary: a direct PyTorch algorithm file operating on a Gymnasium environment. Add real abstractions, such as a replay buffer for DQN, only when the algorithm needs them.
