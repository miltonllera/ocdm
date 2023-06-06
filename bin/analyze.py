import hydra
import pyrootutils
import pytorch_lightning as pl
from omegaconf import DictConfig

from .init import config, utils


root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,  # add to system path
    dotenv=True,      # load environment variables .env file
    cwd=True,         # change cwd to root
)


log = utils.get_logger("bin.analysis")


@hydra.main(
    config_path="../configs",
    config_name="model_analysis.yaml",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    log.info("Analysis starting...")

    pl.seed_everything(cfg.seed)

    analysis_module = config.instantiate_analysis(cfg)

    log.info("Running analysis...")
    analysis_module.run()

    log.info("Analysis completed")

if __name__ == "__main__":
    main()
