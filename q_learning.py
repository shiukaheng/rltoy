import tyro

from td_control import Config, run


if __name__ == "__main__":
    run("Q-learning", tyro.cli(Config))
