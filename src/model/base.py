from abc import ABC, abstractmethod
from typing import Tuple

import torch
import pytorch_lightning as pl
from src.training.initializer import OPTIMIZATION_CONFIG, TrainingInit


class BaseModel(pl.LightningModule, ABC):
    def __init__(
        self,
        training: TrainingInit
    ) -> None:
        super().__init__()
        self.training_init = training

    @abstractmethod
    def forward(self, inputs):
        """
        Computes and chains outputs from all modules in the model.
        """
        raise NotImplementedError

    @abstractmethod
    def _step(self, batch, batch_idx, phase):
        """
        Method to process a batch and perform common computations on the results.

        This method can be used to encapsulate common processing while allowing
        train/validation/test steps to add functionality like computing extra
        metrics or adding visulizations.
        """
        raise NotImplementedError

    def configure_optimizers(self) -> OPTIMIZATION_CONFIG:
        return self.training_init.initialize(self.parameters())

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ):
        return self._step(batch, batch_idx, "train")

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ):
        return self._step(batch, batch_idx, "val")

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ):
        return self._step(batch, batch_idx, "test")
