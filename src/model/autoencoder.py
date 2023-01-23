from typing import Literal, Tuple

import torch
import torch.nn as nn

from src.layers.initialization import weights_init
from src.training.loss import ReconstructionLoss
from .base import BaseModel, OptimizerInit, SchedulerInit


class VariationalAutoEncoder(BaseModel):
    def __init__(self,
        encoder: nn.Sequential,
        latent: nn.Module,
        decoder: nn.Sequential,
        recons_loss: ReconstructionLoss,
        latent_loss: nn.Module,
        optimizer: OptimizerInit,
        scheduler: SchedulerInit = None,
    ) -> None:
        super().__init__(optimizer, scheduler)
        self.encoder = encoder
        self.decoder = decoder
        self.latent = latent
        self.recons_loss = recons_loss
        self.latent_loss = latent_loss

        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self.encoder)
        weights_init(self.decoder)

    def forward(self, inputs):
        h = self.encoder(inputs)
        z, params = self.latent(h)
        recons = self.decoder(z)
        return recons, z, params

    def embed(self, inputs):
        return self.latent(self.encoder(inputs))[0]

    def decode(self, z):
        return self.decoder(z)

    def posterior(self, inputs):
        return self.latent(self.encoder(inputs))[1]

    def _step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        phase: Literal["train", "val", "test"]
    ):
        is_train = phase == "train"
        if is_train and hasattr(self.latent_loss, "update_parameters"):
            self.latent_loss.update_parameters(self.global_step)

        inputs, targets = batch
        recons, z, params = self.forward(inputs)

        recons_loss = self.recons_loss(recons, targets)
        latent_loss = self.latent_loss(z, params)

        loss = recons_loss + latent_loss

        self.log_dict(
            {
                f"{phase}/loss": loss,
                f"{phase}/latent_term": latent_loss,
                f"{phase}/reconstruction_loss": recons_loss
            },
            on_epoch=not is_train,
            on_step=is_train,
            prog_bar=is_train,
            sync_dist=not is_train,
            rank_zero_only=True
        )

        return loss
