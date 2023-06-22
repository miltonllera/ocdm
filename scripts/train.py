import rootutils
import logging
from datetime import datetime
from functools import partial
from typing import Any

import numpy as np
import torch
import lightning.pytorch as pl
import hydra
import wandb
from omegaconf import OmegaConf, DictConfig
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import WandbLogger
from torchvision.utils import make_grid

from src.dataset.datamodule import CompositionTaskDataModule, DisentangledDataModule
from src.training.optim import TrainingInit


rootutils.setup_root(".", cwd=True)

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

instantiate = partial(hydra.utils.instantiate, _convert_='all')
OmegaConf.register_new_resolver(name="get_cls", resolver=lambda cls: hydra.utils.get_class(cls))
OmegaConf.register_new_resolver(name="get_fn", resolver=lambda fn: hydra.utils.get_method(fn))


@hydra.main(config_path="../configs", config_name="train.yaml", version_base="1.3")
def main(cfg: DictConfig) -> dict[str, float]:
    torch.set_float32_matmul_precision('high')
    _logger.info("Starting training...")

    # Set seed
    seed = pl.seed_everything(cfg.get('seed', int(datetime.now().timestamp())))

    # Create components
    _logger.info("Creating datamodule...")
    dataset_cfg: dict[str, Any] = OmegaConf.to_container(cfg.dataset, resolve=True)  # type: ignore

    datataset_cls = hydra.utils.get_class(dataset_cfg.pop('_target_'))  # type: ignore
    if (prediction_type := dataset_cfg.pop('prediction_type', "unsupervised")) != "composition":
        datamodule = DisentangledDataModule(
            datataset_cls, prediction_type=prediction_type, seed=seed, **dataset_cfg,
        )
    else:
        datamodule = CompositionTaskDataModule(datataset_cls, seed=seed, **dataset_cfg)

    # Create TrainingInit from config
    optimizer_init = instantiate(cfg.training.optimizer, _partial_=True)

    if 'schedulers' in cfg.training and cfg.training.schedulers is not None:
        # Handle multiple schedulers
        schedulers = {}
        for name, sched_cfg in cfg.training.schedulers.items():
            schedulers[name] = instantiate(sched_cfg, _partial_=True)

        scheduling_metric = cfg.training.get('scheduling_metric', 'val/loss')
    else:
        schedulers = scheduling_metric = None

    training = TrainingInit(
        optimizer=optimizer_init, schedulers=schedulers, scheduling_metric=scheduling_metric
    )

    _logger.info("Creating model...")
    model = instantiate(cfg.model, training=training)
    # model = torch.compile(model)

    _logger.info("Creating trainer...")
    ckpt_mng = ModelCheckpoint(cfg.trainer.default_root_dir, monitor='val/loss')

    callbacks = [RichProgressBar(), ckpt_mng]
    if not cfg.trainer.fast_dev_run:
        callbacks.append(ExamplePlotsCallback(num_samples=8))

    trainer = instantiate(
        cfg.trainer,
        logger = WandbLogger(
            project='ocdm',
            name=f"{cfg.condition_name}_{cfg.model_name}",
            save_dir=cfg.trainer.default_root_dir,
            config=OmegaConf.to_container(cfg, resolve=True)
        ) if not cfg.trainer.fast_dev_run else False,
        callbacks=callbacks,
        max_epochs=-1,  # NOTE: training steps is defined directly
        # strategy='ddp_find_unused_parameters_true'
    )

    # Train
    _logger.info("Starting training phase...")
    trainer.fit(model, datamodule=datamodule)

    _logger.info("Training finished.")
    train_metrics = {k: float(v) for k, v in trainer.callback_metrics.items() if 'train' in k}

    # Test if requested
    test_metrics = {}
    if cfg.get('test', False):
        _logger.info("Starting test phase...")
        ckpt_path = ckpt_mng.best_model_path if not trainer.fast_dev_run else None
        trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)
        test_metrics = {k: float(v) for k, v in trainer.callback_metrics.items() if 'test' in k}

    _logger.info("Run completed.")
    return {**train_metrics, **test_metrics}


class ExamplePlotsCallback(pl.Callback):
    """
    Logs input–reconstruction comparisons to Weights & Biases at the end of validation epochs.
    Works with any model that implements `forward(x)` returning a reconstruction.
    """

    def __init__(self, num_samples=8, every_n_epochs=1):
        super().__init__()
        self.num_samples = num_samples

    def on_test_end(self, trainer, pl_module):
        device = pl_module.device

        if not hasattr(pl_module, 'reconstruction'):
            return

        def log_reconstructions_from_loader(dataset, split_name):
            if dataset is None:
                return

            x = [dataset[i][0] for i in np.random.choice(len(dataset), size=self.num_samples)]
            x = torch.stack(x).to(device)

            with torch.no_grad():
                assert hasattr(pl_module, 'reconstruction')
                x_hat = pl_module.reconstruction(x)  # type: ignore

            n = min(self.num_samples, len(x))
            comparison = torch.cat([x[:n], x_hat[:n]]).cpu()
            grid = make_grid(comparison, nrow=n, normalize=False).clip(0, 1)
            wandb_img = wandb.Image(grid, caption=f"{split_name} reconstructions")

            trainer.logger.experiment.log(  # type: ignore
                {f"{split_name}_reconstructions": wandb_img}
            )

        # Validation reconstructions
        if hasattr(trainer.datamodule, "val_dataloader"):  # type: ignore
            log_reconstructions_from_loader(trainer.datamodule.val_dataset, "val")  # type: ignore

        # Test reconstructions
        if hasattr(trainer.datamodule, "test_dataloader"):  # type: ignore
            log_reconstructions_from_loader(trainer.datamodule.test_dataset, "test")  # type: ignore


if __name__ == "__main__":
    main()
