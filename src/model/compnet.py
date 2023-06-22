from typing import Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.loss import ReconstructionLoss
from src.layers.composition import CompositionOp
from src.layers.slot import SlotAttention, SlotDecoder

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


class ObjectCentricCompositionNet(BaseModel):
    def __init__(
        self,
        encoder: nn.Sequential,
        slot: SlotAttention,
        decoder: SlotDecoder,
        slot_selector: nn.Sequential,
        composition_op: CompositionOp,
        recons_loss: ReconstructionLoss,
        training: TrainingInit,
    ):
        super().__init__(training)

        self.encoder = encoder
        self.slot = slot
        self.decoder = decoder
        self.slot_selector = slot_selector
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

        slots_og, slots_tr = slots.unflatten(0, (B, Ni)).chunk(Ni, dim=1)
        # slot_masks_og, slot_masks_tr = slots.unflatten(0, (B, Ni)).chunk(Ni, dim=1)

        selected_og, other_og, _ = self.select_slot(slots_og, actions)  # first is the selected one
        selected_tr, _, _ = self.select_slot(slots_tr, actions)

        selected_slot = torch.cat((selected_og, selected_tr), dim=1)

        transformed_slot = self.composition_op(selected_slot, actions)
        new_slots = torch.cat((transformed_slot, other_og), dim=1)

        all_slots = torch.cat((slots, new_slots), dim=0)
        recons = self.decoder(all_slots).unflatten(0, (B, Ni + 1))

        return recons, new_slots

    def select_slot(self, slots, actions):
        actions = actions.unsqueeze(1).expand(-1, self.slot.n_slots, -1)
        slot_and_actions = torch.cat((slots, actions), dim=-1)

        slot_probs = F.softmax(self.slot_selector(slot_and_actions), dim=1)
        sorted_slots, idx = slot_probs.sort(dim=1, descending=True)

        return sorted_slots[:, :1], sorted_slots[:, 1:], idx

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
