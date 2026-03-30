from typing import Literal

import torch
import torch.nn as nn

from src.model.base import BaseModel, TrainingInit
from src.nn.slot import SlotAttention, FigureGroundSegmentation
from src.nn.utils.parsing import create_sequential


class SlotAutoencoder(BaseModel):
    def __init__(
        self,
        input_size: tuple[int, int, int],
        encoder_config: list,
        decoder_config: list,
        training: TrainingInit,
        n_slots: int = 4,
        slot_size: int = 64,
        n_iter: int = 3,
        slot_channels: int = 1,
        slot_hidden_size: int = 128,
        approx_implicit_grad: bool = True,
    ):
        super().__init__(training)
        self.save_hyperparameters()

        self.encoder = create_sequential(input_size, encoder_config)

        with torch.no_grad():
            if isinstance(input_size, int):
                dummy_input = torch.zeros(1, input_size)
            else:
                dummy_input = torch.zeros(1, *input_size)
            encoder_output = self.encoder(dummy_input)
            encoder_output_size = encoder_output.shape[-1]

        self.slot_attention = SlotAttention(
            input_size=encoder_output_size,
            n_slots=n_slots,
            slot_size=slot_size,
            n_iter=n_iter,
            slot_channels=slot_channels,
            hidden_size=slot_hidden_size,
            approx_implicit_grad=approx_implicit_grad
        )

        self.decoder = create_sequential(slot_size, decoder_config)
        self.recons_loss = nn.MSELoss(reduction='sum')

    def decode(
        self, slots: torch.Tensor, attention_weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, n_slots = slots.size()[:2]
        slots_flat = slots.flatten(end_dim=1)
        rgba = self.decoder(slots_flat)
        rgba = rgba.unflatten(0, (batch_size, n_slots))
        slot_recons, slot_mask_logits = torch.tensor_split(rgba, indices=[-1], dim=2)
        slot_masks = torch.softmax(slot_mask_logits, dim=1)
        recons = (slot_masks * slot_recons).sum(dim=1)
        return recons, slot_masks

    def forward(
            self,
            inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(inputs)
        slots, attention_weights = self.slot_attention(h)
        recons, decoder_masks = self.decode(slots, attention_weights)
        return recons, slots, decoder_masks

    def _step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        phase: Literal["train", "val", "test"]
    ) -> torch.Tensor:
        inputs, targets = batch
        targets = 2 * targets - 1

        recons, _, _ = self.forward(inputs)

        loss = self.recons_loss(recons, targets) / len(targets)
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

    def predict(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self(inputs)

    def embed(self, inputs: torch.Tensor) -> torch.Tensor:
        h = self.encoder(inputs)
        slots, _ = self.slot_attention(h)
        return slots

    def reconstruction(self, inputs):
        output = (self.predict(inputs)[0] + 1) / 2
        return output.clip(0, 1)


class FigureGroundAutoencoder(BaseModel):
    def __init__(
        self,
        input_size: tuple[int, int, int],
        encoder_config: list,
        decoder_config: list,
        training: TrainingInit,
        slot_size: int = 64,
        n_iter: int = 3,
        slot_channels: int = 1,
        slot_hidden_size: int = 128,
        approx_implicit_grad: bool = False,
    ):
        super().__init__(training)
        self.save_hyperparameters()

        self.encoder = create_sequential(input_size, encoder_config)

        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_size)
            encoder_output = self.encoder(dummy_input)
            encoder_output_size = encoder_output.shape[-1]

        self.fig_rep = FigureGroundSegmentation(
            input_size=encoder_output_size,
            latent_size=slot_size,
            n_iter=n_iter,
            n_channels=slot_channels,
            hidden_size=slot_hidden_size,
            approx_implicit_grad=approx_implicit_grad
        )

        self.decoder = create_sequential(slot_size, decoder_config)
        self.recons_loss = nn.MSELoss(reduction='sum')

    def decode(
        self,
        fig_reps: torch.Tensor,
        attention_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fig_reps_flat = fig_reps.squeeze(1)
        rgba = self.decoder(fig_reps_flat)
        fig_recons, fig_mask_logits = torch.tensor_split(rgba, indices=[-1], dim=1)
        fig_masks = torch.sigmoid(fig_mask_logits)
        recons = fig_masks * fig_recons
        return recons, fig_masks

    def forward(self, inputs: torch.Tensor):
        h = self.encoder(inputs)
        slots, attention_weights = self.fig_rep(h)
        recons, decoder_masks = self.decode(slots, attention_weights)
        return recons, slots, decoder_masks

    def _step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        phase: Literal["train", "val", "test"]
    ):
        inputs, targets = batch
        recons, _, _ = self.forward(inputs)

        loss = self.recons_loss(recons, targets) / len(targets)

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

    def predict(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self(inputs)

    def embed(self, inputs):
        h = self.encoder(inputs)
        slots, _ = self.fig_rep(h)
        return slots

    def reconstruction(self, inputs):
        output = self.predict(inputs)[0]
        return output.clip(0, 1)
