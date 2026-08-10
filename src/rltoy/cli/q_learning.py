import tyro

from rltoy.algorithms.q_learning import train
from rltoy.cli.common import RunConfig, run


def main() -> None:
    run(train, tyro.cli(RunConfig), "Q-learning")
