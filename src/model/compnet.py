from typing import Tuple, Literal, Union

import torch
import torch.nn as nn

from src.training.loss import ReconstructionLoss
from src.layers.composition import CompositionOp, SlotComposition
from src.layers.slot import SlotAttention, SlotDecoder
from src.layers.fgseg import FigDecoder, FigureGroundSegmentation

from .base import BaseModel, TrainingInit


class CompositionNet(BaseModel):
    def __init__(
        self,
        encoder: nn.Sequential,
        latent: nn.Module,
        composition_op: CompositionOp,
        decoder: nn.Sequential,
        recons_loss: ReconstructionLoss,
        latent_loss: nn.Module,
        training: TrainingInit,
    ):
        super().__init__(training)
        self.encoder = encoder
        self.latent = latent
        self.composition_op = composition_op
        self.decoder = decoder
        self.recons_loss = recons_loss
        self.latent_loss = latent_loss

    @property
    def latent_size(self):
        return self.latent.latent_size

    @property
    def n_actions(self):
        return self.composition_op.n_actions

    def forward(self, inputs):
        inputs, actions = inputs
        B, Ni = inputs.shape[:2]

        # Format inputs so that we have shape (2 * batch_size, input_size)
        # and corresponding reference and transform images follow each other
        inputs = inputs.flatten(0, 1)

        h = self.encoder(inputs)
        z, params = self.latent(h)

        z = z.unflatten(0, (B, Ni))
        z_comp = self.composition_op(z, actions)

        z_s = torch.cat([z, z_comp.unsqueeze(1)], dim=1).contiguous()
        recons = self.decoder(z_s.flatten(0, 1)).unflatten(0, (B, Ni + 1))

        return recons, z_s, params

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


class ForegroundCompNet(BaseModel):
    def __init__(
        self,
        encoder: nn.Sequential,
        slot: Union[SlotAttention, FigureGroundSegmentation],
        decoder: Union[SlotDecoder, FigDecoder],
        composition_op: SlotComposition,
        recons_loss: ReconstructionLoss,
        training: TrainingInit,
    ):
        if not (isinstance(slot, SlotAttention) and isinstance(decoder, SlotDecoder)) and \
            not (isinstance(slot, FigureGroundSegmentation) and isinstance(decoder, FigDecoder)):
                raise ValueError

        super().__init__(training)

        self.encoder = encoder
        self.slot = slot
        self.decoder = decoder
        self.composition_op = composition_op
        self.recons_loss = recons_loss

    def forward(self, inputs):
        inputs, actions = inputs
        B, Ni = inputs.shape[:2]  # Ni is always 2

        # Format inputs so that we have shape (2 * batch_size, input_size)
        # and corresponding reference and transform images follow each other
        inputs = inputs.flatten(0, 1)

        h = self.encoder(inputs)
        slots, _ = self.slot(h)

        # Composition operations expect a shape of (B, Ni, Z). Thus we undo the flattening
        # of the first 2 dimension used to apply the input encoder and squeeze the slot dimension.
        # slots = slots.unflatten(0, (B, Ni))
        new_slots = self.composition_op(
            slots.squeeze(1).unflatten(0, (B, Ni)), actions
        ).unsqueeze(1)

        # We return the outputs in shape (B, Ni, 1, Z), but need to flatten the first two
        # dimension when decoding, so only the unsqueezing is saved.
        all_slots = torch.cat((slots, new_slots), dim=0).unsqueeze(1)
        recons, _ = self.decoder((all_slots.flatten(0, 1), None))

        return recons.unflatten(0, (B, Ni + 1)), all_slots

    def _step(self, batch, batch_idx, phase):
        is_train = phase == "train"

        inputs, targets = batch
        recons, _ = self.forward(inputs)

        recons_loss = self.recons_loss(recons, targets)

        self.log_dict(
            {
                f"{phase}/loss": recons_loss,
            },
            on_epoch=not is_train,
            on_step=is_train,
            prog_bar=is_train,
            sync_dist=not is_train,
            rank_zero_only=True
        )

        return recons_loss
