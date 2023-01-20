import os
import os.path as osp
from functools import partial
from typing import List, Tuple, Dict
from ignite.metrics import Metric

import torch
import hydra
import pytorch_lightning as pl
from hydra.utils import get_class, get_method
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import Logger

from bin.extra.analysis import Analysis
from bin.extra.visualization import Visualzation
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
OmegaConf.register_new_resolver(
    name="get_fn", resolver=lambda fn: get_method(fn)
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
        f"dataset <{dataset_cfg.dataset_cls}>..."
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


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
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
    logger: List[Logger],
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

    metrics: Dict[str, Metric] = instantiate_metrics(cfg.metrics)

    visualizations = instantiate_visualizations(cfg.visualizations)

    logger = instantiate_loggers(cfg.logger)[0]

    analysis_module = Analysis(
        datamodule=datamodule,
        model=model,
        trainer=trainer,
        metrics=metrics,
        visualizations=visualizations,
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


def load_model(run_path: str) -> pl.LightningModule:
    return load_run(run_path)[1]


def instantiate_metrics(
    metric_cfg: DictConfig
) -> Dict[str, Metric]:
    metrics: Dict[str, Metric] = {}

    if not metric_cfg:
        _log.warning("No metric configs found! Skipping...")
        return metrics

    if not isinstance(metric_cfg, DictConfig):
        raise TypeError("Metric config must be a DictConfig!")

    for name, m_conf in metric_cfg.items():
        if isinstance(m_conf, DictConfig) and "_target_" in m_conf:
            _log.info(f"Initializing metric <{m_conf._target_}>")
            metrics[str(name)] = instantiate(m_conf)

    return metrics


def instantiate_visualizations(
    viz_cfg: DictConfig
) -> List[Visualzation]:
    visualizations: List[Visualzation] = []

    if not viz_cfg:
        _log.warning("No visualisations in analysis! Skipping...")
        return visualizations

    for _, v_cfg in viz_cfg.items():
        if isinstance(v_cfg, DictConfig) and "_target_" in v_cfg:
            _log.info(f"Initializing visualization <{v_cfg._target_}>")
            visualizations.append(instantiate(v_cfg))

    return visualizations


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
