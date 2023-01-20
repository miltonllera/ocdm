from typing import Tuple, Literal
import torch
import torch.nn as nn

from src.model.base import BaseModel, TrainingInit
from src.layers.slot import SlotAttention, SlotDecoder
from src.layers.stochastic import DiagonalGaussian
from src.training.loss import ReconstructionLoss


class SlotDecoderControl(BaseModel):
    def __init__(self,
        encoder: nn.Sequential,
        decoder: nn.Sequential,
        slot: SlotAttention,
        loss: ReconstructionLoss,
        training: TrainingInit,
    ):
        super().__init__(training)
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
        slot_size: int,
        encoder: nn.Sequential,
        decoder: SlotDecoder,
        latent: DiagonalGaussian,
        loss: ReconstructionLoss,
        training: TrainingInit,
    ):
        assert latent.size == n_slots * slot_size

        super().__init__(training)
        self.n_slots = n_slots
        self.slot_size = slot_size
        self.encoder = encoder
        self.latent = latent
        self.slot_decoder = decoder
        self.recons_loss = loss

    def forward(self, inputs: torch.Tensor):
        h = self.encoder(inputs)
        z, params = self.latent(h)
        slots = z.unflatten(1, (self.n_slots, self.slot_size))
        recons, slot_masks = self.slot_decoder((slots, None))
        return (recons, slot_masks), (slots, params)

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


class SlotSpatialBroadcast(BaseModel):
    def __init__(
        self,
        n_slots: int,
        slot_size: int,
        encoder: nn.Sequential,
        decoder: nn.Sequential,
        latent: DiagonalGaussian,
        loss: ReconstructionLoss,
        training: TrainingInit,
    ):
        assert latent.size == n_slots * slot_size

        super().__init__(training)
        self.n_slots = n_slots
        self.slot_size = slot_size
        self.encoder = encoder
        self.latent = latent
        self.decoder = decoder
        self.recons_loss = loss

    def forward(self, inputs: torch.Tensor):
        B = len(inputs)

        h = self.encoder(inputs)
        z, params = self.latent(h)

        slots = z.unflatten(1, (self.n_slots, self.slot_size)).flatten(0, 1)
        per_slot_recons = self.decoder(slots).unflatten(0, (B, self.n_slots))

        recons = per_slot_recons.sum(1)

        return (recons, per_slot_recons), (slots, params)

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
