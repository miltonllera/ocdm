from typing import Callable, Dict, Iterable, TypedDict, Optional, Union

from torch.nn.parameter import Parameter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as Scheduler, ChainedScheduler


OptimizerInit = Callable[[Iterable[Parameter]], Optimizer]
SchedulerInit = Callable[[Optimizer], Scheduler]
Schedulers = Optional[Union[SchedulerInit, Dict[str, SchedulerInit]]]


class SCHEDULER_CONFIG(TypedDict):
    scheduler: Scheduler
    interval: str
    monitor: Optional[str]


class OPTIMIZATION_CONFIG(TypedDict, total=False):
    optimizer: Optimizer
    lr_scheduler: SCHEDULER_CONFIG


class TrainingInit:
    def __init__(
        self,
        optimizer: OptimizerInit,
        schedulers: Optional[Schedulers] = None,
        scheduling_metric: Optional[str] = None,
    ):
        assert schedulers is None or scheduling_metric is not None

        self.optimizer_init = optimizer
        self.schedulers = schedulers
        self.schedling_metric = scheduling_metric

    def initialize(self, parameters) -> OPTIMIZATION_CONFIG:
        optimizer = self.optimizer_init(parameters)
        config: OPTIMIZATION_CONFIG = {'optimizer': optimizer}

        if self.schedulers is None:
            return config

        if isinstance(self.schedulers, dict):
            schedulers = list(self.schedulers.values())
            if len(schedulers) == 1:
                schedulers = schedulers[0]
        else:
            schedulers = self.schedulers

        if isinstance(schedulers, list):
            scheduler = ChainedScheduler(
                [si(optimizer) for si in schedulers])
        else:
            scheduler = schedulers(optimizer)

        scheduler_config: SCHEDULER_CONFIG = {
            'scheduler': scheduler,
            'interval': "step",
            'monitor': self.schedling_metric,
        }
        config['lr_scheduler'] = scheduler_config

        return config
