from config import ExperimentConfig
from runner import run_experiment


def main() -> None:
    config = ExperimentConfig.parse()
    run_experiment(config)


if __name__ == "__main__":
    main()