import tyro

from td_control import Config, run


if __name__ == "__main__":
    run("SARSA", tyro.cli(Config))
