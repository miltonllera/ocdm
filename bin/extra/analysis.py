from typing import Callable, Dict, List

import torch
import pytorch_lightning as pl
import pandas as pd
from ignite.engine import create_supervised_evaluator
from ignite.metrics import Metric
from torch.utils.data import default_collate

from .logger import FolderLogger


class Analysis:
    def __init__(
        self,
        datamodule: pl.LightningDataModule,
        model: pl.LightningModule,
        trainer: pl.Trainer,
        metrics: Dict[str, Metric],
        visualizations: List[Callable],
        logger: FolderLogger,
        precompute_outputs: bool = True,
        collate_fn: Callable = default_collate
    ) -> None:
        self.datamodule = datamodule
        self.model = model
        self.trainer = trainer
        self.metrics = metrics
        self.visualisations = visualizations
        self.logger = logger
        self.precompute_outputs = precompute_outputs
        self.collate_fn = collate_fn

        self.initialize()

    def initialize(self):
        for v in self.visualisations:
            if hasattr(v, "set_owner"):
                v.set_owner(self)

    def run(self):
        self.datamodule.setup("fit")
        self.datamodule.setup("predict")

        # if self.precompute_outputs:
        #     train_output = self.compute_outputs(

        if len(self.metrics) > 0:
            self.compute_metrics()

        if len(self.visualisations) > 0:
            self.visualize()

    def compute_metrics(self):
        engine = create_supervised_evaluator(self.model, self.metrics)

        train_metrics = self.score(engine, self.datamodule.val_dataloader())
        test_metrics = self.score(engine, self.datamodule.test_dataloader())

        metrics = pd.concat(
            [train_metrics, test_metrics],
            keys=['Train', 'Test'], names=['Data']
        )

        self.logger.log_metrics("metrics", metrics.reset_index())

    def visualize(self):
        for viz in self.visualisations:
            figure = viz(self.model, self.datamodule)
            self.logger.log_visualisation(figure)


    @torch.no_grad()
    def score(self, engine, loader):
        metrics = engine.run(loader).metrics

        index = pd.Index(metrics.keys(), name='Metric')
        scores = pd.Series(metrics.values(), index=index, name='Score')

        return scores

    @torch.no_grad()
    def compute_outputs(self, loader):
        model = self.model

        model.eval()
        device = next(model.parameters()).device

        outputs, targets = [], []
        for i, (x, t) in enumerate(loader):
            if i >= 10:
                break

            x = x.to(device=device)
            z = model(x)

            outputs.append(z)
            targets.append(t)

        outputs = self.collate_fn(outputs)
        targets = self.collate_fn(targets)

        return outputs, targets
