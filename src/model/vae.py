from abc import abstractmethod
from typing import Callable, Literal

import torch
import torch.nn as nn

from src.nn.init import weights_init
from src.nn.stochastic import DiagonalGaussian
from src.nn.utils.parsing import create_sequential
from src.training.loss import (
    ReconstructionLoss,
    UpdatableLoss,
    GaussianKL,
    WassersteinAdversarial,
    WassersteinMMD
)
from .base import BaseModel, TrainingInit


class VariationalAutoencoder(BaseModel):
    """Base class for variational autoencoders with common encoder/decoder structure."""

    def __init__(
        self,
        input_size: tuple[int, int, int] | int,
        encoder_config: list,
        decoder_config: list,
        training: TrainingInit,
        latent_size: int = 10,
        recons_loss: Literal["bce", "mse", "l1"] = "mse",
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

        self.latent = DiagonalGaussian(encoder_output_size, latent_size)
        self.decoder = create_sequential(latent_size, decoder_config)
        self.recons_loss = ReconstructionLoss(recons_loss)

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

    @abstractmethod
    def _compute_latent_loss(
            self,
            z: torch.Tensor,
            z_params: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        raise NotImplementedError

    def _step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        phase: Literal["train", "val", "test"]
    ) -> torch.Tensor:
        inputs, targets = batch
        recons, z, params = self.forward(inputs)

        recons_loss = self.recons_loss(recons, targets)
        latent_loss = self._compute_latent_loss(z, params)

        loss = recons_loss + latent_loss

        is_train = phase == "train"
        self.log_dict(
            {
                f"{phase}/loss": loss,
                f"{phase}/latent_term": latent_loss,
                f"{phase}/reconstruction_term": recons_loss,
            },
            on_epoch=not is_train,
            on_step=is_train,
            prog_bar=is_train,
            sync_dist=True,
            rank_zero_only=True
        )

        return loss if is_train else recons_loss

    def predict(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward(inputs)[1]

    def reconstruction(self, inputs: torch.Tensor):
        return self.forward(inputs)[0]


class BetaVAE(VariationalAutoencoder):
    """β-VAE using Gaussian KL divergence loss."""

    def __init__(
        self,
        input_size: tuple[int, int, int] | int,
        encoder_config: list,
        decoder_config: list,
        training: TrainingInit,
        latent_size: int = 10,
        beta: float = 1.0,
        beta_schedule: tuple[int, str, float] | None = None,
    ):
        super().__init__(input_size, encoder_config, decoder_config, training, latent_size, "bce")
        self.latent_loss_fn = GaussianKL(beta=beta, beta_schedule=beta_schedule)

    def _compute_latent_loss(
        self,
        z: torch.Tensor,
        z_params: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        return self.latent_loss_fn(z, z_params)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if isinstance(self.latent_loss_fn, UpdatableLoss):
            self.latent_loss_fn.update_parameters(self.global_step)

    def reconstruction(self, inputs: torch.Tensor):
        return super().reconstruction(inputs).sigmoid_()


class WassersteinAE(VariationalAutoencoder):
    """Wasserstein Autoencoder using adversarial discriminator."""

    def __init__(
        self,
        input_size: tuple[int, int, int] | int,
        encoder_config: list,
        decoder_config: list,
        training: TrainingInit,
        latent_size: int = 10,
        use_discrete: bool = True,
        lambda1: float = 10.0,
        lambda2: float = 0.0,
        prior_var: float = 1.0,
        lmbda_schedule: tuple[int, float] | None = None,
        discriminator: nn.Module | None = None,
        optimizer_fn: Callable | None = None,
    ):
        super().__init__(input_size, encoder_config, decoder_config, training, latent_size, "mse")
        self.use_discrete = use_discrete
        self.latent_loss_fn = WassersteinAdversarial(
            lambda1=lambda1,
            lambda2=lambda2,
            prior_var=prior_var,
            lmbda_schedule=lmbda_schedule,
            discriminator=discriminator,
            optimizer=optimizer_fn
        )

    def _compute_latent_loss(
            self,
            z: torch.Tensor,
            z_params: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        if self.use_discrete:
            z = z_params[0]
        return self.latent_loss_fn(z, z_params)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if isinstance(self.latent_loss_fn, UpdatableLoss):
            self.latent_loss_fn.update_parameters(self.global_step)

    def train(self, mode=True):
        result = super().train(mode)
        if hasattr(self.latent_loss_fn, 'train'):
            self.latent_loss_fn.train(mode)
        return result

    def eval(self):
        result = super().eval()
        if hasattr(self.latent_loss_fn, 'eval'):
            self.latent_loss_fn.eval()
        return result


class WassersteinMMDAE(VariationalAutoencoder):
    """Wasserstein Autoencoder using Maximum Mean Discrepancy."""

    def __init__(
        self,
        input_size: tuple[int, int, int] | int,
        encoder_config: list,
        decoder_config: list,
        training: TrainingInit,
        latent_size: int = 10,
        use_discrete: bool = True,
        lambda1: float = 10.0,
        lambda2: float = 1.0,
        prior_type: Literal["norm", "unif"] = "norm",
        prior_var: float = 1.0,
        kernel: Callable | None = None,
        lambda_schedule: tuple[int, float] | None = None,
    ):
        super().__init__(input_size, encoder_config, decoder_config, training, latent_size, "mse")
        self.use_discrete = use_discrete
        self.latent_loss_fn = WassersteinMMD(
            lambda1=lambda1,
            lambda2=lambda2,
            prior_type=prior_type,
            prior_var=prior_var,
            kernel=kernel,
            lambda_schedule=lambda_schedule
        )

    def _compute_latent_loss(
        self,
        z: torch.Tensor,
        z_params: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        if self.use_discrete:
            z = z_params[0]
        return self.latent_loss_fn(z, z_params)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if isinstance(self.latent_loss_fn, UpdatableLoss):
            self.latent_loss_fn.update_parameters(self.global_step)
