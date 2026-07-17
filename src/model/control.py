from typing import Literal

import torch
import torch.nn as nn

from src.model.base import BaseModel, TrainingInit
from src.nn.slot import SlotAttention, FigureGroundSegmentation
from src.nn.stochastic import DiagonalGaussian
from src.nn.utils.parsing import create_sequential
from src.nn.spatial import PositionEmbedding2D
from src.training.loss import WassersteinMMD


class SlotDecoderControl(BaseModel):
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
        approx_implicit_grad: bool = False,
    ):
        super().__init__(training)
        self.save_hyperparameters()

        self.encoder = create_sequential(input_size, encoder_config)

        with torch.no_grad():
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

        decoder_input_size = n_slots * slot_size
        self.decoder = create_sequential(decoder_input_size, decoder_config)
        self.recons_loss = nn.MSELoss(reduction='sum')

    def forward(
        self,
        inputs: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        h = self.encoder(inputs)
        slots, masks = self.slot_attention(h)
        recons = self.decoder(slots.flatten(1))
        return recons, (slots, masks)

    def _step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        phase: Literal["train", "val", "test"]
    ):
        inputs, targets = batch
        targets = 2 * targets - 1
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
        input_size: tuple[int, int, int],
        encoder_config: list,
        decoder_config: list,
        training: TrainingInit,
        n_slots: int = 4,
        slot_size: int = 64,
        beta: float = 1.0,
    ):
        super().__init__(training)
        self.save_hyperparameters()

        self.n_slots = n_slots
        self.slot_size = slot_size

        self.encoder = create_sequential(input_size, encoder_config)

        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_size)
            encoder_output = self.encoder(dummy_input)
            encoder_output_size = encoder_output.shape[-1]

        latent_size = n_slots * slot_size
        self.latent = DiagonalGaussian(encoder_output_size, latent_size)
        self.decoder = create_sequential(slot_size, decoder_config)
        self.recons_loss = nn.MSELoss(reduction='sum')
        self.beta = beta

    def decode(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        h = self.encoder(inputs)
        z, params = self.latent(h)
        slots = z.unflatten(1, (self.n_slots, self.slot_size))
        recons, slot_masks = self.decode(slots)
        return (recons, slot_masks), (slots, params)

    def _step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        phase: Literal["train", "val", "test"]
    ):
        inputs, targets = batch
        targets = 2 * targets - 1

        (recons, _), (_, params) = self.forward(inputs)

        recons_loss = self.recons_loss(recons, targets)

        mu, log_var = params
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1).mean()
        latent_loss = self.beta * kl_loss

        loss = recons_loss + latent_loss

        is_train = phase == "train"
        self.log_dict(
            {
                f"{phase}/loss": loss,
                f"{phase}/latent_term": latent_loss,
                f"{phase}/reconstruction_loss": recons_loss,
                f"{phase}/kl_loss": kl_loss,
            },
            on_epoch=not is_train,
            on_step=is_train,
            prog_bar=is_train,
            sync_dist=not is_train,
            rank_zero_only=True
        )

        return loss

    def predict(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self(inputs)

    def reconstruction(self, inputs):
        output = (self.predict(inputs)[0] + 1) / 2
        return output.clip(0, 1)


class FigureGroundDecoderControl(BaseModel):
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
        use_wasserstein_reg: bool = False,
    ):
        super().__init__(training)
        self.save_hyperparameters()

        self.encoder = create_sequential(input_size, encoder_config)

        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_size)
            encoder_output = self.encoder(dummy_input)
            # encoder_output is (B, C, H, W)
            C, H, W = encoder_output.shape[1:]

        # Shared position embedding at the class level
        self.pos_emb = PositionEmbedding2D(n_channels=slot_size, height=H, width=W, embed='cardinal')

        self.fig_rep = FigureGroundSegmentation(
            input_size=C,
            latent_size=slot_size,
            n_iter=n_iter,
            n_channels=slot_channels,
            hidden_size=slot_hidden_size,
            approx_implicit_grad=approx_implicit_grad
        )

        # Decoder receives slot_size features directly (standard fully-connected + conv transpose)
        self.decoder = create_sequential(slot_size, decoder_config)
        self.recons_loss = nn.MSELoss(reduction='sum')
        if use_wasserstein_reg:
            self.latent_loss_fn = WassersteinMMD(
                lambda1=10.0,
                lambda2=0.0,  # No z_param variance regularization
                prior_type='norm',
                prior_var=2.0,
                kernel=None,
                lambda_schedule=None
            )
        else:
            self.latent_loss_fn = None

    def decode(self, fig_reps: torch.Tensor) -> torch.Tensor:
        fig_reps_flat = fig_reps.squeeze(1) # (B, slot_size)
        recons = self.decoder(fig_reps_flat)
        return recons

    def forward(
            self,
            inputs: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        h = self.encoder(inputs) # (B, C, H, W)

        # Add positional embedding
        pos_info = self.pos_emb.projection(self.pos_emb.grid.to(inputs.device)).permute(2, 0, 1)
        h = h + pos_info

        # Flatten spatial dimensions to (B, H*W, C)
        h_flat = h.flatten(2).transpose(1, 2)

        fig_reps, masks = self.fig_rep(h_flat)
        recons = self.decode(fig_reps)
        return recons, (fig_reps, masks)

    def _step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        phase: Literal["train", "val", "test"]
    ):
        inputs, targets = batch
        targets = 2 * targets - 1

        recons, (fig_rep, _) = self.forward(inputs)

        recons_loss = self.recons_loss(recons, targets) / len(targets)
        if is_train := (phase == "train" and self.latent_loss_fn is not None):
            latent_loss = self.latent_loss_fn(fig_rep, None)  # type: ignore
        else:
            latent_loss = 0.0

        self.log_dict(
            {
                f"{phase}/loss": recons_loss + latent_loss,
                f"{phase}/latent_loss": latent_loss,
            },
            on_epoch=not is_train,
            on_step=is_train,
            prog_bar=is_train,
            sync_dist=not is_train,
            rank_zero_only=True
        )

        return recons_loss + latent_loss

    def predict(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self(inputs)

    def embed(self, inputs: torch.Tensor) -> torch.Tensor:
        h = self.encoder(inputs)
        pos_info = self.pos_emb.projection(self.pos_emb.grid.to(inputs.device)).permute(2, 0, 1)
        h = h + pos_info
        h_flat = h.flatten(2).transpose(1, 2)
        fig_reps, _ = self.fig_rep(h_flat)
        return fig_reps

    def reconstruction(self, inputs):
        output = (self.predict(inputs)[0] + 1) / 2
        return output.clip(0, 1)


class FigureGroundControl(BaseModel):
    def __init__(
        self,
        input_size: tuple[int, int, int],
        encoder_config: list,
        decoder_config: list,
        training: TrainingInit,
        latent_size: int = 64,
        beta: float = 1.0,
    ):
        super().__init__(training)
        self.save_hyperparameters()

        self.latent_size = latent_size

        self.encoder = create_sequential(input_size, encoder_config)

        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_size)
            encoder_output = self.encoder(dummy_input)
            encoder_output_size = encoder_output.shape[-1]

        self.latent = DiagonalGaussian(encoder_output_size, latent_size)
        self.decoder = create_sequential(latent_size, decoder_config)
        self.recons_loss = nn.MSELoss(reduction='sum')
        self.beta = beta

    def decode(self, fig_rep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rgba = self.decoder(fig_rep)
        fig_recons, fig_mask_logits = torch.tensor_split(rgba, indices=[-1], dim=1)
        fig_masks = torch.sigmoid(fig_mask_logits)
        recons = fig_masks * fig_recons
        return recons, fig_masks

    def forward(
        self,
        inputs: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        h = self.encoder(inputs)
        z, params = self.latent(h)
        recons, fig_masks = self.decode(z)
        return (recons, fig_masks), (z, params)

    def _step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        phase: Literal["train", "val", "test"]
    ):
        inputs, targets = batch
        targets = 2 * targets - 1

        (recons, _), (_, params) = self.forward(inputs)

        recons_loss = self.recons_loss(recons, targets) / len(targets)

        mu, log_var = params
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1).mean()
        latent_loss = self.beta * kl_loss

        loss = recons_loss + latent_loss

        is_train = phase == "train"
        self.log_dict(
            {
                f"{phase}/loss": loss,
                f"{phase}/latent_term": latent_loss,
                f"{phase}/reconstruction_loss": recons_loss,
                f"{phase}/kl_loss": kl_loss,
            },
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
        z, _ = self.latent(h)
        return z.unsqueeze(1)

    def reconstruction(self, inputs):
        output = (self.predict(inputs)[0][0] + 1) / 2
        return output.clip(0, 1)
