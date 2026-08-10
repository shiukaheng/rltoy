import tyro

from rltoy.algorithms.sarsa import train
from rltoy.cli.common import RunConfig, run


def main() -> None:
    run(train, tyro.cli(RunConfig), "SARSA")
