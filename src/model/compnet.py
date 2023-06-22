import torch
import torch.nn as nn
from typing import Literal

from src.nn.composition import FixedInterpolationComp
from src.nn.stochastic import DiagonalGaussian
from src.nn.utils.parsing import create_sequential
from src.training.loss import GaussianKL

from .base import BaseModel, TrainingInit


class CompositionNet(BaseModel):
    def __init__(
        self,
        input_size: tuple[int, int, int] | int,
        encoder_config: list,
        decoder_config: list,
        training: TrainingInit,
        latent_size: int = 10,
        n_actions: int = -1,  # -1 means same as latent_size
        beta: float = 1.0,
    ):
        super().__init__(training)
        self.save_hyperparameters()

        # Build encoder from config
        self.encoder = create_sequential(input_size, encoder_config)

        # Get encoder output size for latent layer input
        with torch.no_grad():
            if isinstance(input_size, int):
                dummy_input = torch.zeros(1, input_size)
            else:
                dummy_input = torch.zeros(1, *input_size)
            encoder_output = self.encoder(dummy_input)
            encoder_output_size = encoder_output.shape[-1]

        # Build latent layer (DiagonalGaussian)
        self.latent = DiagonalGaussian(encoder_output_size, latent_size)

        # Build composition operation
        actual_n_actions = latent_size if n_actions == -1 else n_actions
        self.composition_op = FixedInterpolationComp(latent_size, actual_n_actions)

        # Build decoder from config
        self.decoder = create_sequential(latent_size, decoder_config)

        # Fixed losses
        self.recons_loss = nn.MSELoss(reduce='sum')
        self.latent_loss_fn = GaussianKL(beta=beta)

    @property
    def latent_size(self):
        return self.latent.latent_size

    @property
    def n_actions(self):
        return self.composition_op.n_actions

    def forward(self, inputs):
        h = self.encoder(inputs)
        z, params = self.latent(h)
        recons = self.decoder(z)
        return recons, z, params

    def _step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        phase: Literal["val", "test"]
    ) -> torch.Tensor:
        inputs, targets = batch

        assert phase != "train"
        assert not isinstance(inputs, tuple)

        recons, z, params = self.forward(inputs)

        recons_loss = self.recons_loss(recons, targets) / len(inputs)
        latent_loss = self.latent_loss_fn(z, params)
        loss = recons_loss + latent_loss

        self.log_dict(
            {
                f"{phase}/loss": loss,
                f"{phase}/reconstruction_term": recons_loss,
                f"{phase}/latent_term": latent_loss,
            },
            on_epoch=True,
            on_step=False,
            prog_bar=False,
            sync_dist=True,
            rank_zero_only=True
        )

        return recons_loss

    def training_step(
            self, batch: tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor], batch_idx: int
    ):
        (inputs, actions), targets = batch
        B = len(inputs)

        # Format inputs so that we have shape (2 * batch_size, input_size)
        # and corresponding reference and transform images follow each other
        inputs = inputs.flatten(0, 1)

        h = self.encoder(inputs)
        z, params = self.latent(h)

        z = z.unflatten(0, (B, 2))
        z_comp = self.composition_op(z, actions)

        zs = torch.cat([z, z_comp.unsqueeze(1)], dim=1).contiguous()
        recons = self.decoder(zs.flatten(0, 1)).unflatten(0, (B, 3))

        recons_loss = self.recons_loss(recons, targets) / B
        latent_loss = self.latent_loss_fn(zs, params)
        loss = recons_loss + latent_loss

        self.log_dict(
            {
                f"train/loss": loss,
                f"train/latent_term": latent_loss,
                f"train/reconstruction_term": recons_loss,
            },
            on_epoch=False,
            on_step=True,
            prog_bar=True,
            sync_dist=True,
            rank_zero_only=True
        )

        return loss

    def reconstruction(self, inputs: torch.Tensor):
        recons = self.forward(inputs)[0]
        # TODO: apply sigmoid if using VAE formulation.
        return recons

    def predict(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate predictions without gradients."""
        with torch.no_grad():
            return self.forward(inputs)

