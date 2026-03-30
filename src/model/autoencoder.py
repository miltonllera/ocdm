from abc import abstractmethod
from typing import Callable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt

from src.nn.init import weights_init
from src.nn.stochastic import DiagonalGaussian, GumbelSoftmax
from src.nn.embedding import Quantization
from src.nn.discriminator import PatchDiscriminator
from src.nn.utils.parsing import create_sequential
from src.training.loss import (
    DiscriminatorLoss,
    ReconstructionLoss,
    UpdatableLoss,
    GaussianKL,
    WassersteinAdversarial,
    WassersteinMMD
)
from .base import BaseModel, TrainingInit


#------------------------------------ Variational Autoencoders -----------------------------------

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


#-------------------------------------- Discrete Autoencoder -------------------------------------

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

    def forward(self, inputs, hard=None):
        h = self.patch_encoder(inputs).permute(0, 2, 3, 1)
        B, H, W, _ = h.shape
        weights, _ = self.latent(h, hard=hard)
        z_q = weights.flatten(0, 2) @ self.feature_dict
        recons = self.patch_decoder(z_q.unflatten(0, (B, H, W)).permute(0, 3, 1, 2))
        return recons, z_q, weights

    def embed(self, inputs, hard=None, reshape='undo'):
        h = self.patch_encoder(inputs).permute(0, 2, 3, 1)
        weights = self.latent(h, hard=hard)[0]
        z_q = weights.flatten(0, 2) @ self.feature_dict

        B, _, H, W = h.shape
        if reshape == 'undo':
            z_q = z_q.unflatten(0, (B, H, W)).permute(0, 3, 1, 2)
        elif reshape == 'tokenization':
            z_q = z_q.unflatten(0, (B, H * W))

        return z_q

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


#---------------------------------- Vector Quantized Autoencoder ---------------------------------

class VectorQuantizedAutoencoder(BaseModel):
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
        beta: float = 0.25,
    ):
        super().__init__(training)
        self.save_hyperparameters()

        # Build patch encoder from config
        H, W = resolution
        self.resolution = H, W
        self.patch_encoder = create_sequential(input_size, encoder_config)

        # Get patch encoder output size for latent layer input
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_size)
            patch_output = self.patch_encoder(dummy_input)
            assert patch_output.shape[-2:] == (H, W)
            patch_output_size = patch_output.shape[-3]

        # Build quantization latent layer
        self.latent_proj = nn.Linear(patch_output_size, token_dim)
        self.feature_codebook = Quantization(vocab_size, token_dim, beta)

        # Build patch decoder from config
        decoder_input_size = token_dim, H, W
        self.patch_decoder = create_sequential(decoder_input_size, decoder_config)

        self.recons_loss = ReconstructionLoss()
        self.reset_parameters()

    def reset_parameters(self):
        self.feature_codebook.reset_parameters()
        weights_init(self.patch_encoder)
        weights_init(self.patch_decoder)

    def forward(self, inputs):
        h = self.patch_encoder(inputs)
        B, _, H, W = h.shape

        z =  self.latent_proj(h.permute(0, 2, 3, 1).flatten(0, 2))
        z_q, idx, dist = self.feature_codebook(z)
        z_q = z_q.unflatten(0, (B, H, W)).permute(0, 3, 1, 2)
        idx = idx.unflatten(0, (B, H, W))
        dist = dist.unflatten(0, (B, H, W))

        recons = self.patch_decoder(z_q)
        return recons, idx, dist

    def embed(self, inputs, reshape='tokenization'):
        h = self.patch_encoder(inputs)
        z =  self.latent_proj(h.permute(0, 2, 3, 1).flatten(0, 2))
        z_q, idx, _ = self.feature_codebook(z)

        B, _, H, W = h.shape
        if reshape == 'undo':
            z_q = z_q.unflatten(0, (B, H, W)).permute(0, 3, 1, 2)
        elif reshape == 'tokenization':
            z_q = z_q.unflatten(0, (B, H * W))

        return z_q

    def decode(self, emb):
        _, S = emb.shape[:2]
        assert S == self.resolution[0] * self.resolution[1]
        if emb.dtype == torch.long:
            z_q = self.feature_codebook(emb)
        else:
            z_q = emb
        return self.patch_decoder(z_q.unflatten(1, (self.resolution)).permute(0, 3, 1, 2))

    def reconstruction(self, inputs: torch.Tensor):
        return self.forward(inputs)[0]

    def _step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs, targets = batch
        recons, z_q, dist = self.forward(inputs)
        recons_loss = self.recons_loss(recons, targets)
        codebook_loss = dist.sum() / len(inputs)
        return recons_loss, codebook_loss, z_q

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        inputs, targets = batch
        recons_loss, codebook_loss, _ = self._step(batch, batch_idx)
        loss = recons_loss + codebook_loss

        self.log_dict(
            {
                f"train/loss": loss,
                f"train/recons_loss": recons_loss,
                f"train/codebook_loss": codebook_loss,
            },
            on_epoch=False,
            on_step=True,
            prog_bar=True,
            sync_dist=True,
            rank_zero_only=True
        )

        return loss

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        phase: Literal["val", "test"] = "val",
    ) -> torch.Tensor:
        recons_loss, codebook_loss, _ = self._step(batch, batch_idx)
        self.log_dict(
            {
                f"{phase}/loss": recons_loss,
                f"{phase}/codebook_loss": codebook_loss,
            },
            on_epoch=True,
            on_step=False,
            prog_bar=False,
            sync_dist=True,
            rank_zero_only=True
        )
        return recons_loss

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        return self.validation_step(batch, batch_idx, "test")


class VectorQuantizedGAN(VectorQuantizedAutoencoder):
    def __init__(
        self,
        input_size: tuple[int, int, int],
        training: TrainingInit,
        encoder_config: list,
        decoder_config: list,
        resolution: tuple[int, int] = (16, 16),
        vocab_size: int = 256,
        token_dim: int = 64,
        beta: float = 0.25,
        discriminator_layers: int = 3,
        discriminator_first_layer_channels: int = 64,
        discriminator_learning_rate: float = 0.0001,
        discriminator_penalty_warmup: int = 10_000,
    ):
        super().__init__(
            input_size,
            training,
            encoder_config,
            decoder_config,
            resolution,
            vocab_size,
            token_dim,
            beta
        )

        self.discriminator_warmup = discriminator_penalty_warmup
        self.discriminator_lr = discriminator_learning_rate
        self.discriminator = PatchDiscriminator(
            discriminator_layers, discriminator_first_layer_channels
        )
        self.disc_loss = DiscriminatorLoss('hinge')
        self.automatic_optimization = False

    def configure_optimizers(self):
        vqae_params = (
            list(self.patch_encoder.parameters()) +
            list(self.patch_decoder.parameters()) +
            list(self.latent_proj.parameters()) +
            list(self.feature_codebook.parameters())
        )
        disc_params = list(self.discriminator.parameters())
        config = self.training_init.initialize(vqae_params)
        vq_ae_opt = config['optimizer']
        disc_opt = opt.Adam(disc_params, self.discriminator_lr, betas=[0.5, 0.9])

        return [vq_ae_opt, disc_opt], [config['scheduler']] if 'scheduler' in config else []

    def _step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs, targets = batch
        recons, z_q, dist = self.forward(inputs)
        recons_loss = self.recons_loss(recons, targets)
        codebook_loss = dist.sum() / len(inputs)
        disc_logits = self.discriminator(recons)
        if self.disc_loss.loss_type == 'bce':
            disc_loss = F.binary_cross_entropy_with_logits(
                disc_logits, torch.ones_like(disc_logits)
            )
        else:
            disc_loss = -torch.mean(disc_logits)
        return recons_loss, disc_loss, codebook_loss, recons, z_q

    def _update_discriminator(self, recons, targets):
        logits_real = self.discriminator(targets)
        logits_recons = self.discriminator(recons.detach())
        return self.disc_loss(logits_recons, logits_real)

    def _compute_discriminator_loss_weight(self, rec_loss, disc_loss):
        if self.trainer.global_step < self.discriminator_warmup:
            delta = torch.tensor([0.0], device=disc_loss.device)
        last_layer = self.patch_decoder[-1].weight
        r_grad = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]
        d_grad = torch.autograd.grad(disc_loss, last_layer, retain_graph=True)[0]
        delta = torch.norm(r_grad) / (torch.norm(d_grad) + 1e-6)
        return torch.clamp(delta, 0.0, 1e4).detach()

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        vq_ae_opt, disc_opt = self.optimizers()

        # optimize VQAE
        vq_ae_opt.zero_grad()
        recons_loss, adv_loss, codebook_loss, recons, z_q = self._step(batch, batch_idx)
        delta = self._compute_discriminator_loss_weight(recons_loss, adv_loss)
        loss = recons_loss + codebook_loss + delta * adv_loss
        self.manual_backward(loss)
        vq_ae_opt.step()

        # optimize GAN discriminator
        disc_opt.zero_grad()
        _, targets = batch
        disc_loss = self._update_discriminator(recons, targets)
        self.manual_backward(disc_loss)
        disc_opt.step()

        self.log_dict(
            {
                'train/disc_loss':  disc_loss,
                "train/loss": loss,
                "train/recons_loss": recons_loss,
                "train/codebook_loss": codebook_loss,
                "train/adv_loss": adv_loss,
            },
            on_epoch=False,
            on_step=True,
            prog_bar=True,
            sync_dist=True,
            rank_zero_only=True
        )

        return loss

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        phase: Literal["val", "test"] = "val",
    ) -> torch.Tensor:
        _, targets = batch
        recons_loss, adv_loss, codebook_loss, recons, _ = self._step(batch, batch_idx)
        disc_loss = self._update_discriminator(recons, targets)

        metrics = {
            f"{phase}/loss": recons_loss,
            f"{phase}/codebook_loss": codebook_loss,
            f"{phase}/adv_loss": adv_loss,
            f"{phase}/disc_loss": disc_loss,
        }

        self.log_dict(
            metrics,
            on_epoch=True,
            on_step=False,
            prog_bar=False,
            sync_dist=True,
            rank_zero_only=True
        )

        return recons_loss + adv_loss

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        return self.validation_step(batch, batch_idx, "test")

