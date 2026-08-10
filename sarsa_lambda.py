import tyro

from td_control import LambdaConfig, run


if __name__ == "__main__":
    run("SARSA(lambda)", tyro.cli(LambdaConfig))
