import os.path as osp
from functools import partial
from typing import List, Tuple

import hydra
import pytorch_lightning as pl
from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import LightningLoggerBase

from .utils import get_logger


_log = get_logger(__name__)

INSTANTIATED_RUN_MODULES =  Tuple[
    pl.LightningDataModule,
    pl.LightningModule,
    pl.Trainer
]

# Always convert to base python classes
instantiate = partial(hydra.utils.instantiate, _convert_="all")

# Add custom resolvers
OmegaConf.register_new_resolver(name="sum", resolver=lambda x, y: x + y)
OmegaConf.register_new_resolver(name="prod", resolver=lambda x, y: x * y)
OmegaConf.register_new_resolver(
    name="get_cls", resolver=lambda cls: get_class(cls)
)


#-------------------------------- Runs ----------------------------------------

def instantiate_run(cfg) -> INSTANTIATED_RUN_MODULES:
    datamodule = instantiate_datamodule(cfg.dataset)
    model = instantiate_model(cfg.model)
    callbacks = instantiate_callbacks(cfg.callbacks)
    loggers = instantiate_loggers(cfg.logger)
    trainer = instantiate_trainer(cfg.trainer, callbacks, loggers)
    return datamodule, model, trainer


def instantiate_datamodule(dataset_cfg) -> pl.LightningDataModule:
    _log.info(
        f"Initializing datamodule <{dataset_cfg._target_}> with "
        f"dataset <{dataset_cfg.dataset_name}>..."
    )
    datamodule: pl.LightningDataModule = instantiate(dataset_cfg)
    return datamodule


def instantiate_model(model_cfg) -> pl.LightningModule:
    _log.info(f"Initializing model <{model_cfg._target_}>...")
    model: pl.LightningModule = instantiate(model_cfg)
    return model


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config."""
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        _log.warning("No callback configs found! Skipping...")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            _log.info(f"Initializing callback <{cb_conf._target_}>")
            callbacks.append(instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[LightningLoggerBase]:
    """Instantiates loggers from config."""
    logger: List[LightningLoggerBase] = []

    if not logger_cfg:
        _log.warning("No logger configs found! Make sure you are not debugging.")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            _log.info(f"Initializing logger <{lg_conf._target_}>")
            logger.append(instantiate(lg_conf))

    return logger


def instantiate_trainer(
    trainer_cfg: DictConfig,
    callbacks: List[Callback],
    logger: List[LightningLoggerBase],
) -> pl.Trainer:
    _log.info(f"Initializing trainer <{trainer_cfg._target_}>...")

    trainer: pl.Trainer = instantiate(
        trainer_cfg,
        callbacks= callbacks,
        logger=logger,
    )

    return trainer


def load_run(run_path: str) -> DictConfig:
    assert osp.isdir(run_path), f"Run log directory {run_path} does not exist"

    _log.info(f"Recomposing run found at <{run_path}>...")

    config_path = osp.join(run_path, ".hydra", "hydra.yaml")
    overrides_path = osp.join(run_path, ".hydra", "overrides.yaml")

    loaded_config = OmegaConf.load(config_path).hydra.job.config_name
    overrides = OmegaConf.load(overrides_path)

    return hydra.compose(loaded_config, overrides=overrides)
