from typing import Literal, Tuple

import torch
import torch.nn as nn

from src.model.base import BaseModel, TrainingInit
from src.layers.slot import SlotAttention, SlotDecoder
from src.layers.fgseg import FigDecoder, FigureGroundSegmentation
from src.training.loss import ReconstructionLoss


class SlotAutoEncoder(BaseModel):
    def __init__(
        self,
        encoder: nn.Sequential,
        decoder: SlotDecoder,
        slot: SlotAttention,
        loss: ReconstructionLoss,
        training: TrainingInit,
    ):
        super().__init__(training)
        self.encoder = encoder
        self.slot_atten = slot
        self.slot_decoder = decoder
        self.recons_loss = loss

    def forward(self, inputs: torch.Tensor):
        h = self.encoder(inputs)
        slots = self.slot_atten(h)
        recons, decoder_masks = self.slot_decoder(slots)
        return (recons, decoder_masks), slots

    def _step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        phase: Literal["train", "val", "test"]
    ):
        inputs, targets = batch
        (recons, _), (_, _) = self.forward(inputs)

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

    def embed(self, inputs):
        h = self.encoder(inputs)
        return self.slot_atten(h)[0]


class FigGroundAutoEncoder(BaseModel):
    def __init__(
        self,
        encoder: nn.Sequential,
        decoder: FigDecoder,
        fig_rep: FigureGroundSegmentation,
        loss: ReconstructionLoss,
        training: TrainingInit,
    ):
        super().__init__(training)
        self.encoder = encoder
        self.fig_rep = fig_rep
        self.fig_decoder = decoder
        self.recons_loss = loss

    def forward(self, inputs: torch.Tensor):
        h = self.encoder(inputs)
        slots = self.fig_rep(h)
        recons, decoder_masks = self.fig_decoder(slots)
        return (recons, decoder_masks), slots

    def _step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        phase: Literal["train", "val", "test"]
    ):
        inputs, targets = batch
        (recons, _), (_, _) = self.forward(inputs)

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

    def embed(self, inputs):
        h = self.encoder(inputs)
        return self.fig_rep(h)[0]
