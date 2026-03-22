from abc import abstractmethod
from typing import Callable, Literal

import torch
import torch.nn as nn

from src.nn.init import weights_init
from src.nn.stochastic import DiagonalGaussian, GumbelSoftmax
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
        # Latent space parameters
        latent_size: int = 10,
        # Reconstruction loss parameters
        recons_loss: Literal["bce", "mse", "l1"] = "mse",
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

        # Build latent layer (DiagonalGaussian) with explicit parameters
        self.latent = DiagonalGaussian(encoder_output_size, latent_size)

        # Build decoder from config
        self.decoder = create_sequential(latent_size, decoder_config)

        # Reconstruction loss
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
        """Compute the latent space loss (KL, MMD, adversarial, etc.)"""
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
        """Generate predictions without gradients."""
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
        # Latent space parameters
        latent_size: int = 10,
        # KL loss parameters
        beta: float = 1.0,
        beta_schedule: tuple[int, str, float] | None = None,
    ):
        super().__init__(input_size, encoder_config, decoder_config, training, latent_size, "bce")

        # KL divergence loss
        self.latent_loss_fn = GaussianKL(beta=beta, beta_schedule=beta_schedule)

    def _compute_latent_loss(
        self,
        z: torch.Tensor,
        z_params: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        kl_loss = self.latent_loss_fn(z, z_params)
        # Log additional metrics for β-VAE
        # self.log("kl_loss", kl_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        # self.log("beta", self.latent_loss_fn.beta, on_step=True, on_epoch=False, prog_bar=False)
        # if hasattr(self.latent_loss_fn, 'anneal'):
        #     self.log(
        #         "anneal",
        #         self.latent_loss_fn.anneal,
        #         on_step=True,
        #         on_epoch=False,
        #         prog_bar=False
        #     )

        return kl_loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update β schedule if applicable."""
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
        # Latent space parameters
        latent_size: int = 10,
        use_discrete: bool = True,
        # Wasserstein adversarial loss parameters
        lambda1: float = 10.0,
        lambda2: float = 0.0,
        prior_var: float = 1.0,
        lmbda_schedule: tuple[int, float] | None = None,
        discriminator: nn.Module | None = None,
        optimizer_fn: Callable | None = None,
    ):
        super().__init__(input_size, encoder_config, decoder_config, training, latent_size, "mse")

        # Wasserstein adversarial loss
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
            z = z_params[0]  # discrete latents just use a mean-field approximation
        adv_loss = self.latent_loss_fn(z, z_params)

        # Log additional metrics
        # self.log("adversarial_term", adv_loss, on_step=True, on_epoch=True, prog_bar=False)
        # self.log(
        #     "lambda1", self.latent_loss_fn.lambda1, on_step=True, on_epoch=False, prog_bar=False
        # )
        # if hasattr(self.latent_loss_fn, 'anneal'):
        #     self.log(
        #         "anneal",
        #         self.latent_loss_fn.anneal,
        #         on_step=True,
        #         on_epoch=False,
        #         prog_bar=False,
        #     )

        return adv_loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update discriminator and λ schedule if applicable."""
        if isinstance(self.latent_loss_fn, UpdatableLoss):
            self.latent_loss_fn.update_parameters(self.global_step)

    def train(self, mode=True):
        """Override to handle discriminator training mode."""
        result = super().train(mode)
        if hasattr(self.latent_loss_fn, 'train'):
            self.latent_loss_fn.train(mode)
        return result

    def eval(self):
        """Override to handle discriminator eval mode."""
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
        # Latent space parameters
        latent_size: int = 10,
        use_discrete: bool = True,
        # Wasserstein MMD loss parameters
        lambda1: float = 10.0,
        lambda2: float = 1.0,
        prior_type: Literal["norm", "unif"] = "norm",
        prior_var: float = 1.0,
        kernel: Callable | None = None,
        lambda_schedule: tuple[int, float] | None = None,
    ):
        super().__init__(input_size, encoder_config, decoder_config, training, latent_size, "mse")

        # Wasserstein MMD loss
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
            z = z_params[0]  # discrete latents just use a mean-field approximation
        mmd_loss = self.latent_loss_fn(z, z_params)

        # Log additional metrics
        # self.log(
        #     "mmd_term", mmd_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True
        # )
        # self.log(
        #     "lambda1", self.latent_loss_fn.lambda1, on_step=True, on_epoch=False, prog_bar=False
        # )
        # self.log(
        #     "lambda2", self.latent_loss_fn.lambda2, on_step=True, on_epoch=False, prog_bar=False
        # )
        # if hasattr(self.latent_loss_fn, 'anneal'):
        #     self.log(
        #         "anneal",
        #         self.latent_loss_fn.anneal,
        #         on_step=True,
        #         on_epoch=False,
        #         prog_bar=False,
        #     )

        return mmd_loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update λ schedule if applicable."""
        if isinstance(self.latent_loss_fn, UpdatableLoss):
            self.latent_loss_fn.update_parameters(self.global_step)


class DiscreteAutoencoder(BaseModel):
    def __init__(
        self,
        input_size: tuple[int, int, int],
        training: TrainingInit,
        # Encoder-decoder
        encoder_config: list,
        decoder_config: list,
        # latent
        resolution: tuple[int, int] = (16, 16),
        vocab_size: int = 256,
        token_dim: int = 64,
        tau: float = 1.0,
        tau_start: float | None = None,
        tau_steps: float | None = None,
    ):
        super().__init__(training)
        self.save_hyperparameters()

        # Build patch encoder from config
        H, W = resolution
        self.patch_encoder = create_sequential(input_size, encoder_config)

        # Get patch encoder output size for latent layer input
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_size)
            patch_output = self.patch_encoder(dummy_input)
            assert patch_output.shape[-2:] == (H, W)
            patch_output_size = patch_output.shape[-3]

        # Build GumbelSoftmax latent layer
        self.latent = GumbelSoftmax(
            input_size=patch_output_size,
            n_cat=vocab_size,
            tau=tau,
            tau_start=tau_start,
            tau_steps=tau_steps
        )
        self.feature_dict = nn.Parameter(torch.empty(vocab_size, token_dim))
        # Build patch decoder from config
        decoder_input_size = token_dim, H, W
        self.patch_decoder = create_sequential(decoder_input_size, decoder_config)

        self.recons_loss = ReconstructionLoss()
        self.reset_parameters()

    def reset_parameters(self):
        self.feature_dict.data.uniform_(-0.1, 0.1)
        weights_init(self.patch_encoder)
        weights_init(self.patch_decoder)

    def forward(self, inputs):
        h = self.patch_encoder(inputs).permute(0, 2, 3, 1)
        B, H, W, _ = h.shape
        z, logits = self.latent(h)
        features = z.flatten(0, 2) @ self.feature_dict
        recons = self.patch_decoder(features.unflatten(0, (B, H, W)).permute(0, 3, 1, 2))
        return recons, z, logits

    def embed(self, inputs):
        return self.latent(self.patch_encoder(inputs))[0]

    def decode(self, z):
        B, _, H, W = z.shape
        features = z.flatten(2) @ self.feature_dict
        return self.patch_decoder(features.unflatten(0, (B, H, W)).permute(0, 3, 1, 2))

    def _step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        phase: Literal["train", "val", "test"]
    ) -> torch.Tensor:
        inputs, targets = batch
        recons, _, _ = self.forward(inputs)
        loss = self.recons_loss(recons, targets)

        is_train = phase == "train"
        self.log_dict(
            {
                f"{phase}/loss": loss,
            },
            on_epoch=not is_train,
            on_step=is_train,
            prog_bar=is_train,
            sync_dist=True,
            rank_zero_only=True
        )

        return loss

    def reconstruction(self, inputs: torch.Tensor):
        return self.forward(inputs)[0]
