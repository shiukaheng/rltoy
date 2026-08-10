# rltoy Contributor Guide

## Purpose

`rltoy` is a pedagogical reinforcement-learning repository. Its primary product is readable, executable algorithm code, not a general RL framework. Optimize changes for a learner who opens one file, follows the update rule, modifies it, and runs it directly.

## Non-Negotiable Design Rules

- Keep each algorithm's training loop, action selection, trajectory collection, and update rule in its own module under `src/rltoy/algorithms/`.
- Do not introduce registries, base learner classes, generic agent objects, callback frameworks, or a training framework unless a concrete feature cannot be implemented without one.
- Share only mechanics that are genuinely algorithm-independent. `algorithms/tabular.py` intentionally contains schedules, epsilon-greedy selection, snapshots, and result records, not learning updates.
- Prefer a small amount of duplication over hiding an algorithm's defining operation behind an abstraction.
- Every bundled learner must be directly executable as `uv run src/rltoy/algorithms/<file>.py`.
- Use standard Gymnasium `reset()` and `step()` APIs. Do not create a project-specific environment interface.
- Keep Pygame and rendering code out of learner logic. Learners publish snapshots through the optional `observer` argument.

## Repository Map

- `src/rltoy/algorithms/`: standalone learners. The algorithm file is the primary teaching artifact.
- `src/rltoy/algorithms/tabular.py`: small shared tabular mechanics and observer data types.
- `src/rltoy/envs/graph_world.py`: JSON-authored finite graph MDP and Gymnasium implementation.
- `src/rltoy/envs/data/branching_risk.json`: bundled registered graph environment.
- `src/rltoy/visualization/graph_renderer.py`: GraphWorld-only Pygame renderer and observer adapter.
- `src/rltoy/cli/common.py`: command-line environment and renderer plumbing. Keep it out of update rules.
- `src/rltoy/experiments/compare_tabular.py`: exact-planner comparison for tabular TD-control learners only.
- `tests/`: focused behavior tests. Add a test alongside every new learner or renderer contract.
- `pyproject.toml`: dependencies and optional installed command aliases.

## Running And Verifying

Use `uv`; do not use bare `pip` or modify the virtual environment manually.

```bash
uv sync
uv run pytest
uv build
```

Smoke-test a learner from its file:

```bash
uv run src/rltoy/algorithms/q_learning.py --episodes 2
SDL_VIDEODRIVER=dummy uv run src/rltoy/algorithms/q_learning.py --episodes 1 --render --render-delay-ms 0
```

The installed Gymnasium checker may emit an `np.bool8` deprecation warning on CLI runs. It originates in Gymnasium, not project code.

## Add A Learner

1. Create one readable module in `src/rltoy/algorithms/`. Use a descriptive name such as `actor_critic.py`.
2. Define a frozen configuration dataclass in that file. Keep algorithm-specific parameters there. Do not force neural algorithms to inherit tabular learning-rate or epsilon settings.
3. Implement `train(env, config, seed=None, observer=observe_nothing)`. Validate the observation/action spaces before allocating tables or networks.
4. Seed NumPy or PyTorch where the algorithm uses them. Preserve Gymnasium's first-episode reset seed convention used by existing learners.
5. Return `TrainingResult`. For value learners, its `q_values` field contains action values. For policy learners, it currently contains action probabilities for CLI reporting; do not treat it as an actual Q-table. A critic's per-episode estimates belong in `state_values`.
6. Publish `TrainingSnapshot` values after updates when visualization is useful:
   - `action_values`: shape `(states, actions)` for Q-like learners. The renderer derives `V(s) = max_a Q(s, a)` and a greedy policy.
   - `state_values`: shape `(states,)` for a critic or planner.
   - `action_probabilities`: shape `(states, actions)` for a stochastic policy. The renderer displays them on outgoing action edges.
   - A snapshot needs at least one of these. A learner with a critic and policy should publish both `state_values` and `action_probabilities`.
7. Add a local `main()` with Tyro and an `if __name__ == "__main__": main()` footer. Define a local run-config subclass with `environment_id`, `graph_path`, `seed`, and render fields, following the neural learner examples.
8. Add an optional command alias in `[project.scripts]` only for bundled learners. A copied experimental file must not require a new alias.
9. Add focused tests in `tests/test_<algorithm>.py`. At minimum, use a deterministic one-step `GraphWorldEnv`, assert returns and output shapes, and ensure values/probabilities are finite. Test mathematical helpers directly when they encode a core rule, such as GAE recursion.
10. Update the README's learner table and runnable examples.

## Add Or Change An Environment

- `GraphWorldEnv` is discrete by design. Preserve the required JSON fields: `start_state`, `actions`, and `states`.
- Every nonterminal state must implement every declared action. Every action outcome list must be nonempty and probabilities must sum to one.
- Terminal states have no declared actions; the environment provides their absorbing transitions internally for planning.
- Keep display metadata optional. Use `position`, labels, action key bindings, and state display colors only for rendering, not MDP semantics.
- Preserve `env.P` because exact planning and comparison scripts rely on it.
- Add validation and Gymnasium behavior tests when changing graph parsing or transitions.

## Visualization Contract

- `GraphRenderer` is specific to `GraphWorldEnv`; do not make it a required part of an environment or learner.
- Value-based display: nodes use the lava ramp and show `V=...`.
- Policy-based display: outgoing action edges use the lava ramp, width, and `pi=...` labels. Policy probability belongs to the action edge, not the node.
- Actor-critic display can use both simultaneously. The on-screen key must accurately describe active color meanings.
- Preserve cyan trajectory highlighting and green/red terminal outcome colors.
- Renderer tests should use a recording renderer plus `pygame.time.wait` monkeypatch rather than opening a Pygame window.

## Dependencies And Scope

- Add Python dependencies with `uv add <package>` so `pyproject.toml` and `uv.lock` stay reproducible.
- Do not add PyTorch helpers, replay buffers, normalization utilities, or shared neural modules preemptively. Add a local implementation when an algorithm needs it, then extract only after clear repeated use.
- Do not add compatibility layers or broad refactors unless there is a concrete consumer or failure to address.
- Keep comments short and explanatory. Prefer code that mirrors textbook pseudocode.

## Before Finishing

Run the focused learner tests, then `uv run pytest`, `uv build`, and a headless render smoke test when snapshots or rendering changed. Check `git diff --check`. Do not commit or push unless explicitly requested.
