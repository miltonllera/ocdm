from abc import ABC, abstractmethod
from typing import Callable, Iterable, TypedDict, Optional

import pytorch_lightning as pl
from torch.nn.parameter import Parameter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as Scheduler


OptimizerInit = Callable[[Iterable[Parameter]], Optimizer]
SchedulerInit = Optional[Callable[[Optimizer], Scheduler]]


class SchedulerConfig(TypedDict):
    scheduler: Scheduler
    interval: str
    monitor: Optional[str]


class OptimizationConfig(TypedDict, total=False):
    optimizer: Optimizer
    lr_scheduler: SchedulerConfig


class BaseModel(pl.LightningModule, ABC):
    def __init__(
        self,
        optimizer: OptimizerInit,
        scheduler: SchedulerInit = None,
        scheduler_metric: Optional[str] = "train/loss",
    ) -> None:
        super().__init__()
        self.optimizer_init = optimizer
        self.scheduler_init = scheduler
        self.scheduler_metric = scheduler_metric

    @abstractmethod
    def forward(self, inputs):
        pass

    # @abstractmethod
    # def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
    #     pass

    def configure_optimizers(self):
        optimizer = self.optimizer_init(self.parameters())
        config: OptimizationConfig = {'optimizer': optimizer}

        if self.scheduler_init is not None:
            scheduler = self.scheduler_init(optimizer)
            scheduler_config: SchedulerConfig = {
                'scheduler': scheduler,
                'interval': "step",
                'monitor': self.scheduler_metric,
            }
            config['lr_scheduler'] = scheduler_config

        return config
