# rltoy

`rltoy` is a small laboratory for learning reinforcement learning by reading and changing working implementations.

Its design rule is simple: an algorithm's training loop and update rule live together in one readable file. Shared code is only for mechanics that are not the algorithm itself, such as epsilon-greedy action selection, schedules, and result recording.

## Run

Install the project with `uv sync`, then run a learner directly from its file:

```bash
uv run src/rltoy/algorithms/q_learning.py --episodes 1000
uv run src/rltoy/algorithms/deep_q_learning.py --episodes 1000
uv run src/rltoy/algorithms/actor_critic.py --episodes 1000
uv run src/rltoy/algorithms/gae.py --episodes 1000 --gae-lambda 0.95
uv run src/rltoy/algorithms/reinforce.py --episodes 1000
uv run src/rltoy/algorithms/reinforce_baseline.py --episodes 1000
uv run src/rltoy/algorithms/sarsa.py --episodes 1000
uv run src/rltoy/algorithms/sarsa_lambda.py --episodes 1000 --trace-decay 0.8
```

Animate a learner on the bundled graph:

```bash
uv run src/rltoy/algorithms/q_learning.py --render --render-delay-ms 50
uv run rltoy-interactive-graph
```

Copy `q_learning.py` to start an experiment. The copied file is immediately runnable; change its `train` loop, then run `uv run src/rltoy/algorithms/my_algorithm.py`. No algorithm registry or project configuration edit is required. The `rltoy-q-learning`, `rltoy-deep-q-learning`, `rltoy-actor-critic`, `rltoy-gae`, `rltoy-reinforce`, `rltoy-reinforce-baseline`, `rltoy-sarsa`, and `rltoy-sarsa-lambda` commands remain optional aliases for the bundled files.

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
src/rltoy/algorithms/deep_q_learning.py # full Deep Q-learning (DQN) loop
src/rltoy/algorithms/actor_critic.py    # full one-step TD actor-critic loop
src/rltoy/algorithms/gae.py             # full GAE actor-critic loop
src/rltoy/algorithms/reinforce.py        # full vanilla REINFORCE loop
src/rltoy/algorithms/reinforce_baseline.py # full REINFORCE-with-baseline loop
src/rltoy/algorithms/sarsa.py           # full SARSA loop
src/rltoy/algorithms/sarsa_lambda.py    # full accumulating-trace SARSA(lambda) loop
src/rltoy/envs/graph_world.py           # JSON graph MDP and Gymnasium interface
src/rltoy/visualization/graph_renderer.py
```

Pygame visualization observes training snapshots but never enters learner logic. Value-based learners color nodes with a lava ramp and label them with `V=...`; policy-only learners such as REINFORCE use that ramp on labeled, weighted outgoing action edges. REINFORCE with a baseline displays both: value colors on nodes and policy colors on edges. The on-screen key identifies the active meanings. The neural learners are direct PyTorch algorithm files operating on discrete Gymnasium environments. Deep Q-learning has a local replay buffer because DQN needs one; vanilla REINFORCE uses only each completed episode and no baseline.
