from dataclasses import dataclass

import tyro

from rltoy.algorithms.sarsa_lambda import train
from rltoy.cli.common import RunConfig, run


@dataclass(frozen=True)
class Config(RunConfig):
    trace_decay: float = 0.8


def main() -> None:
    config = tyro.cli(Config)
    run(train, config, "SARSA(lambda)", config)
