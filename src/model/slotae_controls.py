from typing import Tuple, Literal
import torch
import torch.nn as nn
from layers.stochastic import DiagonalGaussian

from src.model.base import BaseModel, Optimizer, Schedulers
from src.layers.slot import SlotAttention, SlotDecoder
from src.training.loss import ReconstructionLoss


class SlotDecoderControl(BaseModel):
    def __init__(self,
        encoder: nn.Sequential,
        decoder: nn.Sequential,
        slot: SlotAttention,
        loss: ReconstructionLoss,
        optimizer: Optimizer,
        scheduler: Schedulers = None,
    ):
        super().__init__(optimizer, scheduler)
        self.encoder = encoder
        self.slot_atten = slot
        self.decoder = decoder
        self.recons_loss = loss

    def forward(self, inputs: torch.Tensor):
        h = self.encoder(inputs)
        slots, masks = self.slot_atten(h)
        recons = self.decoder(slots.flatten(1))
        return recons, (slots, masks)

    def _step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        phase: Literal["train", "val", "test"]
    ):
        inputs, targets = batch
        recons, (_, _) = self.forward(inputs)

        loss = self.recons_loss(recons, targets)

        is_train = phase == "train"
        self.log(
            f"{phase}/loss",
            loss,
            on_epoch=not is_train,
            on_step=is_train,
            prog_bar=is_train,
            sync_dist=not is_train,
            rank_zero_only=True
        )

        return loss


class SlotAttentionControl(BaseModel):
    def __init__(
        self,
        n_slots: int,
        encoder: nn.Sequential,
        decoder: SlotDecoder,
        latent: DiagonalGaussian,
        loss: ReconstructionLoss,
        optimizer: Optimizer,
        scheduler: Schedulers = None,
    ):
        assert latent.size % n_slots == 0

        super().__init__(optimizer, scheduler)
        self.encoder = encoder
        self.latent = latent
        self.slot_decoder = decoder
        self.recons_loss = loss

    def forward(self, inputs: torch.Tensor):
        h = self.encoder(inputs)
        z = self.latent(h)
        slots = z.unflatten(1, (self.n_slots, z.size(1) // self.n_slots))
        recons, slot_masks = self.slot_decoder(slots)
        return (recons, slot_masks), slots

    def _step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        phase: Literal["train", "val", "test"]
    ):
        inputs, targets = batch
        (recons, _), _ = self.forward(inputs)

        loss = self.recons_loss(recons, targets)

        is_train = phase == "train"
        self.log(
            f"{phase}/loss",
            loss,
            on_epoch=not is_train,
            on_step=is_train,
            prog_bar=is_train,
            sync_dist=not is_train,
            rank_zero_only=True
        )

        return loss
