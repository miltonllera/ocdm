import os
import os.path as osp
from functools import partial
from typing import Any, List, Tuple, Dict

import torch
import hydra
import pytorch_lightning as pl
from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import LightningLoggerBase

from bin.extra.analysis_module import Analysis
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


#----------------------------- Analysis ---------------------------------------

def instantiate_analysis(cfg: DictConfig) -> Analysis:
    _log.info(f"Initializing analysis module...")

    datamodule, model, trainer = load_run(cfg.run_path)

    evaluations = instantiate_evaluators(cfg.evaluators)

    visualisations = instantiate_visualizations(cfg.visualisatoins)

    logger = instantiate_loggers(cfg.logger)

    analysis_module: Analysis = instantiate(
        evaluations=evaluations,
        visualisations=visualisations,
        datamodule=datamodule,
        model=model,
        trainer=trainer,
        logger=logger,
    )

    return analysis_module


def load_run(run_path: str) -> INSTANTIATED_RUN_MODULES:
    _log.info(f"Loading run found at <{run_path}>...")

    run_cfg = load_cfg(run_path)
    run_cfg.logger = None
    datamodule, model, trainer = instantiate_run(run_cfg)
    model.load_state_dict(load_best_model(run_path))

    return datamodule, model, trainer


def instantiate_evaluators(eval_cfg: DictConfig):
    evaluators: List[Any] = []

    if not eval_cfg:
        _log.warning("No evaluator configs found! Skipping...")
        return evaluators

    if not isinstance(eval_cfg, DictConfig):
        raise TypeError("Evaluator config must be a DictConfig!")

    for _, e_conf in eval_cfg.items():
        if isinstance(e_conf, DictConfig) and "_target_" in e_conf:
            _log.info(f"Initializing callback <{e_conf._target_}>")
            evaluators.append(instantiate(e_conf))



def instantiate_visualizations(viz_cfg: DictConfig):
    if not viz_cfg:
        _log.warning("No visualisations in analysis! Skipping...")
        return None
    raise NotImplementedError()


def load_cfg(run_path: str) -> DictConfig:
    assert osp.isdir(run_path), f"Run log directory {run_path} does not exist"

    config_path = osp.join(run_path, ".hydra", "hydra.yaml")
    overrides_path = osp.join(run_path, ".hydra", "overrides.yaml")

    loaded_config = OmegaConf.load(config_path).hydra.job.config_name
    overrides = OmegaConf.load(overrides_path)

    return hydra.compose(loaded_config, overrides=overrides)


def load_best_model(run_path: str) -> Dict[str, torch.Tensor]:
    checkpoint_folder = osp.join(run_path, "checkpoints")
    best_checkpoint = find_best_by_epoch(checkpoint_folder)
    return torch.load(best_checkpoint)['state_dict']


def find_best_by_epoch(checkpoint_folder: str) -> str:
    ckpt_files = os.listdir(checkpoint_folder)  # list of strings

    # checkpoint format is 'epoch_{int}.ckpt'
    def is_epoch_ckpt(f: str):
        return f[6:-5].isdigit()

    best_epoch = max([f[6:-5] for f in ckpt_files if is_epoch_ckpt(f)])

    return osp.join(checkpoint_folder, f"epoch_{best_epoch}.ckpt")
