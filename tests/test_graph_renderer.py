import numpy as np

from rltoy.algorithms.tabular import TrainingSnapshot
from rltoy.visualization.graph_renderer import LAVA_COLORS, GraphTrainingObserver, lava_color


class RecordingRenderer:
    def __init__(self):
        self.calls = []

    def set_state_values(self, values):
        self.calls.append(("values", np.asarray(values)))

    def set_policy_probabilities(self, probabilities):
        self.calls.append(("probabilities", np.asarray(probabilities)))

    def set_policy(self, policy):
        self.calls.append(("policy", np.asarray(policy)))

    def set_trajectory(self, trajectory):
        self.calls.append(("trajectory", trajectory))

    def render(self, state):
        self.calls.append(("render", state))


def test_lava_color_interpolates_between_dark_and_bright_endpoints():
    assert lava_color(0.0) == LAVA_COLORS[0]
    assert lava_color(1.0) == LAVA_COLORS[-1]
    assert lava_color(0.5) not in {LAVA_COLORS[0], LAVA_COLORS[-1]}


def test_observer_renders_action_values_as_state_values(monkeypatch):
    monkeypatch.setattr("pygame.time.wait", lambda _: None)
    renderer = RecordingRenderer()
    observer = GraphTrainingObserver(renderer, delay_ms=0)

    observer(
        TrainingSnapshot(
            0,
            1,
            (0, 1),
            action_values=np.array([[1.0, 3.0], [2.0, 0.0]]),
        )
    )

    assert renderer.calls[0][0] == "values"
    assert np.array_equal(renderer.calls[0][1], [3.0, 2.0])
    assert renderer.calls[1][0] == "policy"
    assert np.array_equal(renderer.calls[1][1], [1, 0])


def test_observer_renders_policy_probabilities_on_edges(monkeypatch):
    monkeypatch.setattr("pygame.time.wait", lambda _: None)
    renderer = RecordingRenderer()
    observer = GraphTrainingObserver(renderer, delay_ms=0)

    observer(
        TrainingSnapshot(
            0,
            1,
            (0, 1),
            action_probabilities=np.array([[0.25, 0.75], [0.5, 0.5]]),
        )
    )

    assert renderer.calls[0][0] == "probabilities"
    assert np.array_equal(renderer.calls[0][1], [[0.25, 0.75], [0.5, 0.5]])


def test_observer_renders_value_baseline_and_policy_edges_together(monkeypatch):
    monkeypatch.setattr("pygame.time.wait", lambda _: None)
    renderer = RecordingRenderer()
    observer = GraphTrainingObserver(renderer, delay_ms=0)

    observer(
        TrainingSnapshot(
            0,
            1,
            (0, 1),
            state_values=np.array([1.5, 2.0]),
            action_probabilities=np.array([[0.25, 0.75], [0.5, 0.5]]),
        )
    )

    assert renderer.calls[0][0] == "values"
    assert np.array_equal(renderer.calls[0][1], [1.5, 2.0])
    assert renderer.calls[1][0] == "probabilities"
    assert np.array_equal(renderer.calls[1][1], [[0.25, 0.75], [0.5, 0.5]])
