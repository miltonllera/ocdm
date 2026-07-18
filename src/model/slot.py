from typing import Literal

import torch
import torch.nn as nn

from src.model.base import BaseModel, TrainingInit
from src.nn.slot import SlotAttention, FigureGroundSegmentation
from src.nn.spatial import PositionEmbedding2D
from src.training.loss import WassersteinMMD
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
        use_wasserstein_reg: bool = False
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

        recons, slots, _ = self.forward(inputs)

        recons_loss = self.recons_loss(recons, targets) / len(targets)
        if is_train := phase == "train" and self.latent_loss_fn is not None:
            latent_loss = self.latent_loss_fn(slots, None)  # type: ignore
        else:
            latent_loss = 0.0

        self.log_dict(
            {
                f"{phase}/loss": recons_loss,
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
        use_wasserstein_reg: bool = False
    ):
        super().__init__(training)
        self.save_hyperparameters()

        self.encoder = create_sequential(input_size, encoder_config)

        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_size)
            encoder_output = self.encoder(dummy_input)
            C, H, W = encoder_output.shape[1:]

        self.pos_emb = PositionEmbedding2D(n_channels=C, height=H, width=W, embed='cardinal')
        self.embedding_mlp = nn.Sequential(
            nn.LayerNorm(C),
            nn.Linear(C, 4 * C),
            nn.ReLU()
        )
        self.decoder_layer_norm = nn.LayerNorm(slot_size)

        self.fig_rep = FigureGroundSegmentation(
            input_size=4 * C,
            latent_size=slot_size,
            n_iter=n_iter,
            n_channels=slot_channels,
            hidden_size=slot_hidden_size,
            approx_implicit_grad=approx_implicit_grad
        )

        self.decoder = create_sequential((slot_size, H, W), decoder_config)
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

    def decode(
        self,
        fig_rep: torch.Tensor,
        attention_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        H, W = self.pos_emb.height, self.pos_emb.width

        fig_rep_broadcast = fig_rep.unsqueeze(1).expand(-1, H, W, -1)
        norm_pos_emb = self.decoder_layer_norm(self.pos_emb.get_projection(fig_rep.device))
        fig_rep_broadcast = fig_rep_broadcast + norm_pos_emb

        rgba = self.decoder(fig_rep_broadcast.permute(0, 3, 1, 2))
        fig_recons, fig_mask_logits = torch.tensor_split(rgba, indices=[-1], dim=1)
        fig_masks = torch.sigmoid(fig_mask_logits)
        recons = fig_masks * fig_recons
        return recons, fig_masks

    def forward(self, inputs: torch.Tensor):
        h = self.encoder(inputs).permute(0, 2, 3, 1)
        h = self.embedding_mlp(self.pos_emb(h).flatten(1, 2))
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
        recons, fig_rep, _ = self.forward(inputs)

        recons_loss = self.recons_loss(recons, targets) / len(targets)
        if is_train := phase == "train" and self.latent_loss_fn is not None:
            latent_loss = self.latent_loss_fn(fig_rep, None)  # type: ignore
        else:
            latent_loss = 0.0

        self.log_dict(
            {
                f"{phase}/loss": recons_loss,
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

    def embed(self, inputs):
        h = self.encoder(inputs)
        slots, _ = self.fig_rep(h)
        return slots

    def reconstruction(self, inputs):
        output = self.predict(inputs)[0]
        return output.clip(0, 1)
