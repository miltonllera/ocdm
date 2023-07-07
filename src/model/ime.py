import copy
from typing import List, Literal, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.modules.container import ModuleList

from src.layers.initialization import weights_init
from src.model.base import BaseModel, TrainingInit


class IndependentMechanismEncoder(BaseModel):
    def __init__(self,
        n_mechanisms: int,
        n_targets: List[int],
        target_type: List[Literal["cat", "reg"]],
        mechanism_encoder: nn.Sequential,
        hidden_size: int,
        encoder_backbone: nn.Sequential,
        # regression_loss: nn.Module,
        # categorical_loss: nn.Module,
        training: TrainingInit,
        mechanism_names: Optional[List[str]] = None,
    ) -> None:
        super().__init__(training)

        assert n_mechanisms > 0
        assert len(n_targets) == n_mechanisms == len(target_type)

        mechanism_predictors = []
        for nt in n_targets:
            mech_enc = copy.deepcopy(mechanism_encoder)
            mech_enc.append(nn.Linear(hidden_size, nt))
            mechanism_predictors.append(mech_enc)

        if mechanism_names is None:
            mechanism_names = [f"mechanism_{i}" for i in range(n_mechanisms)]

        self.target_type = target_type
        self.encoder_backbone = encoder_backbone
        self.mechanism_predictors = ModuleList(mechanism_predictors)
        self.mechanism_names = mechanism_names
        self.regression_loss = nn.MSELoss()
        self.categorical_loss = nn.CrossEntropyLoss()

        self.reset_parameters()

    @property
    def n_mechanisms(self):
        return len(self.target_type)

    def reset_parameters(self):
        weights_init(self.encoder_backbone)
        self.mechanism_predictors.apply(weights_init)

    def forward(self, inputs):
        h = self.encoder_backbone(inputs)
        return [mech(h) for mech in self.mechanism_predictors]

    def _step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        phase: Literal["train", "val", "test"]
    ):

        inputs, targets = batch
        predictions = self.forward(inputs)

        losses, total_loss = {}, 0.0
        for i, pred in enumerate(predictions):
            y = targets[:, i:i + 1]

            if self.target_type[i] == "cat":
                loss = self.categorical_loss(pred, y.long())
            else:
                loss = self.regression_loss(pred, y)

            losses[f'{phase}/{self.mechanism_names[i]}_loss'] = loss
            total_loss = total_loss + loss

        losses[f'{phase}/loss'] =  total_loss

        is_train = phase == "train"
        self.log_dict(
            losses,
            on_epoch=not is_train,
            on_step=is_train,
            prog_bar=is_train,
            sync_dist=not is_train,
            rank_zero_only=True
        )

        return total_loss

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
