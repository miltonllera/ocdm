from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt

from src.nn.init import weights_init
from src.nn.stochastic import GumbelSoftmax
from src.nn.quantization import Quantization
from src.nn.discriminator import PatchDiscriminator
from src.nn.utils.parsing import create_sequential
from src.training.loss import (
    DiscriminatorLoss,
    ReconstructionLoss,
)
from .base import BaseModel, TrainingInit


class DiscreteAutoencoder(BaseModel):
    def __init__(
        self,
        input_size: tuple[int, int, int],
        training: TrainingInit,
        encoder_config: list,
        decoder_config: list,
        resolution: tuple[int, int] = (16, 16),
        vocab_size: int = 256,
        token_dim: int = 64,
        tau: float = 1.0,
        tau_start: float | None = None,
        tau_steps: float | None = None,
    ):
        super().__init__(training)
        self.save_hyperparameters()

        H, W = resolution
        self.patch_encoder = create_sequential(input_size, encoder_config)

        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_size)
            patch_output = self.patch_encoder(dummy_input)
            assert patch_output.shape[-2:] == (H, W)
            patch_output_size = patch_output.shape[-3]

        self.latent = GumbelSoftmax(
            input_size=patch_output_size,
            n_cat=vocab_size,
            tau=tau,
            tau_start=tau_start,
            tau_steps=tau_steps
        )
        self.feature_dict = nn.Parameter(torch.empty(vocab_size, token_dim))

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

    def get_emb_idx(self, z):
        weights, _ = self.latent(z, hard=True)
        return weights.argmax(-1)

    def embed(self, inputs, hard=None, reshape='undo'):
        h = self.patch_encoder(inputs).permute(0, 2, 3, 1)
        weights = self.latent(h, hard=hard)[0]
        z_q = weights.flatten(0, 2) @ self.feature_dict

        B, _, H, W = h.shape
        if reshape == 'undo':
            z_q = z_q.unflatten(0, (B, H, W)).permute(0, 3, 1, 2)
            weights = weights.unflatten(0, (B, H, W))
        elif reshape == 'tokenization':
            z_q = z_q.unflatten(0, (B, H * W))
            weights = weights.unflatten(0, (B, H, W))

        return z_q, weights.argmax(-1)

    def get_quantization(self, z, from_idx):
        if len(z.shape) == 3:
            z = z.argmax(-1)
        return self.feature_dict[z.flatten(0, 1)], z

    def decode(self, z, from_idx: bool = True):
        B, S, H, W  = *z.shape[:2], *self.hparams.resolution  # type: ignore
        assert S == H * W
        features = self.get_quantization(z, from_idx)[0]
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
            {f"{phase}/loss": loss},
            on_epoch=not is_train,
            on_step=is_train,
            prog_bar=is_train,
            sync_dist=True,
            rank_zero_only=True
        )

        return loss

    def reconstruction(self, inputs: torch.Tensor):
        return self.forward(inputs)[0]


class VectorQuantizedAutoencoder(BaseModel):
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
    ):
        super().__init__(training)
        self.save_hyperparameters()

        H, W = resolution
        self.resolution = H, W
        self.patch_encoder = create_sequential(input_size, encoder_config)

        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_size)
            patch_output = self.patch_encoder(dummy_input)
            assert patch_output.shape[-2:] == (H, W)
            patch_output_size = patch_output.shape[-3]

        self.latent_proj = nn.Linear(patch_output_size, token_dim)
        self.feature_codebook = Quantization(vocab_size, token_dim, beta)

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
        z = self.latent_proj(h.permute(0, 2, 3, 1).flatten(0, 2))
        z_q, idx, dist = self.feature_codebook(z)
        recons = self.patch_decoder(z_q.unflatten(0, (B, H, W)).permute(0, 3, 1, 2))
        return recons, idx.unflatten(0, (B, H, W)), dist.unflatten(0, (B, H, W))

    def embed(self, inputs, reshape='tokenization'):
        h = self.patch_encoder(inputs)
        z = self.latent_proj(h.permute(0, 2, 3, 1).flatten(0, 2))
        z_q, idx, _ = self.feature_codebook(z)

        B, _, H, W = h.shape
        if reshape == 'undo':
            z_q = z_q.unflatten(0, (B, H, W)).permute(0, 3, 1, 2)
            idx = idx.unflatten(0, (B, H, W))
        elif reshape == 'tokenization':
            z_q = z_q.unflatten(0, (B, H * W))
            idx = idx.unflatten(0, (B, H * W))

        return z_q, idx

    def get_quantization(self, z, from_idx: bool):
        if from_idx:
            # assume onehot or logits if 3 dimensional matrix
            z_idx = z.argmax(-1) if len(z.shape) == 3 else z
            features = self.feature_codebook.codebook(z_idx)
        else:
            B, S = z.shape[:2]
            features, z_idx, _ = self.feature_codebook(z.flatten(0, 1))
            features, z_idx = features.unflatten(0, (B, S)), z_idx.unflatten(0, (B, S))

        return features, z_idx

    def decode(self, z, from_idx: bool = True):
        features = self.get_quantization(z, from_idx)[0]
        return self.patch_decoder(features.permute(0, 2, 1).unflatten(2, self.hparams.resolution))

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
        recons_loss, codebook_loss, _ = self._step(batch, batch_idx)
        loss = recons_loss + codebook_loss

        self.log_dict(
            {
                "train/loss": loss,
                "train/recons_loss": recons_loss,
                "train/codebook_loss": codebook_loss,
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
            input_size, training, encoder_config, decoder_config,
            resolution, vocab_size, token_dim, beta
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
        vq_ae_opt = config['optimizer']  # type: ignore
        disc_opt = opt.Adam(disc_params, self.discriminator_lr, betas=[0.5, 0.9])  # type: ignore

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
        r_grad = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]  # type: ignore
        d_grad = torch.autograd.grad(disc_loss, last_layer, retain_graph=True)[0]  # type: ignore
        delta = torch.norm(r_grad) / (torch.norm(d_grad) + 1e-6)
        return torch.clamp(delta, 0.0, 1e4).detach()

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        vq_ae_opt, disc_opt = self.optimizers()  # type: ignore

        vq_ae_opt.zero_grad()
        recons_loss, adv_loss, codebook_loss, recons, z_q = self._step(batch, batch_idx)
        delta = self._compute_discriminator_loss_weight(recons_loss, adv_loss)
        loss = recons_loss + codebook_loss + delta * adv_loss
        self.manual_backward(loss)
        vq_ae_opt.step()

        disc_opt.zero_grad()
        _, targets = batch
        disc_loss = self._update_discriminator(recons, targets)
        self.manual_backward(disc_loss)
        disc_opt.step()

        self.log_dict(
            {
                'train/disc_loss': disc_loss,
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

        self.log_dict(
            {
                f"{phase}/loss": recons_loss,
                f"{phase}/codebook_loss": codebook_loss,
                f"{phase}/adv_loss": adv_loss,
                f"{phase}/disc_loss": disc_loss,
            },
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
