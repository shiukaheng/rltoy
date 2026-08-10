# rltoy

`rltoy` is a small reinforcement-learning laboratory for people who want to learn by reading, running, and changing complete implementations.

It favors visible algorithms over reusable framework machinery. Each learner keeps its training loop and update rule in one file. Shared code is limited to mechanics that are not the algorithm itself: Gymnasium environments, schedules, result records, command-line wiring, and optional visualization.

## Start Here

Requirements: Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run src/rltoy/algorithms/monte_carlo.py --episodes 1000
uv run src/rltoy/algorithms/q_learning.py --episodes 1000
```

That command trains tabular Q-learning on the bundled `RLToy/BranchingRisk-v0` environment. Every learner can be run directly from its source file, so the source you read is the code that runs.

To watch learning on the graph:

```bash
uv run src/rltoy/algorithms/q_learning.py --episodes 1000 --render --render-delay-ms 50
```

Use `--help` on any learner to see its configurable hyperparameters:

```bash
uv run src/rltoy/algorithms/gae.py --help
```

## Choose A Learner

| Learner | File | Core idea |
| --- | --- | --- |
| Monte Carlo control | `monte_carlo.py` | First-visit returns from complete episodes, then epsilon-greedy improvement. |
| Q-learning | `q_learning.py` | Off-policy tabular TD control with a max-Q target. |
| SARSA | `sarsa.py` | On-policy tabular TD control using the next sampled action. |
| SARSA(lambda) | `sarsa_lambda.py` | SARSA with accumulating eligibility traces. |
| Deep Q-learning | `deep_q_learning.py` | DQN with a neural Q-network, replay, and target network. |
| REINFORCE | `reinforce.py` | Monte Carlo policy gradient without a baseline. |
| REINFORCE with baseline | `reinforce_baseline.py` | Monte Carlo policy gradient with a learned `V(s)` baseline. |
| Actor-critic | `actor_critic.py` | Online one-step TD error updates for actor and critic. |
| GAE | `gae.py` | Trajectory policy gradients using generalized advantage estimation. |

Run any one directly:

```bash
uv run src/rltoy/algorithms/monte_carlo.py --episodes 1000
uv run src/rltoy/algorithms/sarsa.py --episodes 1000
uv run src/rltoy/algorithms/sarsa_lambda.py --episodes 1000 --trace-decay 0.8
uv run src/rltoy/algorithms/deep_q_learning.py --episodes 1000
uv run src/rltoy/algorithms/reinforce.py --episodes 1000
uv run src/rltoy/algorithms/reinforce_baseline.py --episodes 1000
uv run src/rltoy/algorithms/actor_critic.py --episodes 1000
uv run src/rltoy/algorithms/gae.py --episodes 1000 --gae-lambda 0.95
```

The `rltoy-monte-carlo`, `rltoy-q-learning`, `rltoy-sarsa`, `rltoy-sarsa-lambda`, `rltoy-deep-q-learning`, `rltoy-reinforce`, `rltoy-reinforce-baseline`, `rltoy-actor-critic`, and `rltoy-gae` commands are equivalent installed aliases.

## Read The Visualization

Visualization is optional and lives outside the learning algorithms.

- Value-based learners color graph nodes on a dark-purple-to-yellow lava ramp and show `V=...`.
- Policy-only learners show `pi=...` on outgoing action edges; color and width represent action probability.
- Learners with both a critic and policy, such as actor-critic and GAE, show values on nodes and policy probabilities on edges at once.
- A cyan edge marks the most recent trajectory. Terminal success and failure colors remain green and red.

Run the graph manually with arrow/action key bindings from its JSON metadata:

```bash
uv run rltoy-interactive-graph
```

## Environments

Learners use the standard Gymnasium API:

```python
import gymnasium as gym
import rltoy

env = gym.make("RLToy/BranchingRisk-v0")
state, info = env.reset(seed=0)
next_state, reward, terminated, truncated, info = env.step(0)
```

`GraphWorldEnv` is a finite discrete graph MDP authored as JSON. The bundled graph is at `src/rltoy/envs/data/branching_risk.json`. Load your own graph with:

```python
from rltoy.envs import GraphWorldEnv

env = GraphWorldEnv.from_json("my_graph.json")
```

A graph needs `start_state`, `actions`, and `states`. Every nonterminal state must define every action. Each action has one or more outcomes with `probability`, `next_state`, and optional `reward`. Outcome probabilities for an action must sum to one. Labels, positions, key bindings, and display colors are optional renderer metadata.

The tabular learners require finite `gym.spaces.Discrete` observations and actions. The neural learners currently use one-hot encodings of the same discrete state space. Wrap continuing environments with Gymnasium's `TimeLimit`; do not add a project-specific step-limit API.

## Experiment Safely

The intended way to explore an algorithm is to copy its file and edit it:

```bash
cp src/rltoy/algorithms/q_learning.py src/rltoy/algorithms/my_q_learning.py
uv run src/rltoy/algorithms/my_q_learning.py --episodes 1000
```

No registry or package configuration change is needed for copied experiments. Keep the environment interface and optional observer argument so the copied learner remains runnable and renderable.

To compare the three tabular TD-control implementations against exact value iteration on the graph:

```bash
uv run rltoy-compare-tabular --episodes 500 --runs 20
```

## Layout

```text
src/rltoy/algorithms/      Readable algorithm implementations and small shared mechanics
src/rltoy/envs/            GraphWorldEnv and bundled graph JSON
src/rltoy/visualization/   Pygame graph renderer and training observer
src/rltoy/cli/             Environment, rendering, and direct-run wiring
src/rltoy/experiments/     Comparison scripts
tests/                     Focused environment, learner, and renderer tests
```

## Develop

```bash
uv run pytest
uv build
```

See [`AGENTS.md`](AGENTS.md) for the project design rules and the checklist for adding a new environment or learner.
