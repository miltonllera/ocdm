from typing import Any, Callable, List

import pytorch_lightning as pl
import torchmetrics as tm
import matplotlib.pyplot as plt

from .figure_logger import FigureLogger


class Analysis:
    def __init__(
        self,
        datamodule: pl.LightningDataModule,
        model: pl.LightningModule,
        trainer: pl.Trainer,
        metrics: tm.MetricCollection,
        visualisations: List[Callable[[Any], List[plt.Figure]]],
        logger: FigureLogger,
    ) -> None:
        self.datamodule = datamodule
        self.model = model
        self.trainer = trainer
        self.metrics = metrics
        self.visualisations = visualisations
        self.logger = logger

    def run(self):
        test_data = self.datamodule.test_dataloader()
