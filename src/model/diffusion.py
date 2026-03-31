from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.base import BaseModel, TrainingInit
from src.nn.slot import SlotAttention
from src.nn.spatial import PositionEmbedding2D
from src.nn.diffusion import cosine_schedule, linear_schedule, DiffusionDenoiser


class SlotDiffusion(BaseModel):
    """
    Latent diffusion conditioned on slots.
    """

    def __init__(
        self,
        backbone_type: Literal["dae", "vqvae", "vqgan"],
        backbone_checkpoint: str,
        training: TrainingInit,
        n_slots: int = 4,
        slot_size: int = 192,
        slot_n_iter: int = 3,
        slot_channels: int = 1,
        slot_hidden_size: int = 128,
        slot_approx_implicit_grad: bool = True,
        noise_schedule: str = 'cosine',
        noise_schedule_steps: int = 1000,
        noise_schedule_betas: tuple[float, float] = (0.0, 1.0),
        n_head: int = 4,
        num_layers: int = 4,
        ffwd_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__(training)
        self.save_hyperparameters()

        backbone = SlotDiffusion._load_backbone(backbone_type, backbone_checkpoint)
        backbone.requires_grad_(False)
        self.backbone = backbone
        self.backbone_type = backbone_type

        token_dim = backbone.hparams.token_dim  # type: ignore
        H, W = self.resolution = tuple(backbone.hparams.resolution)  # type: ignore
        self.pos_emb = PositionEmbedding2D(token_dim, H, W, 'cardinal')

        self.slot = SlotAttention(
            input_size=token_dim,
            n_slots=n_slots,
            slot_size=slot_size,
            n_iter=slot_n_iter,
            slot_channels=slot_channels,
            hidden_size=slot_hidden_size,
            approx_implicit_grad=slot_approx_implicit_grad
        )

        self.slot_proj = nn.Linear(slot_size, token_dim, bias=False)
        self.denoiser = DiffusionDenoiser(
            spatial_size=(H, W),
            d_model=token_dim,
            n_head=4,
            num_layers=num_layers,
            ffwd_dim=ffwd_dim,
            dropout=dropout,
        )

        self.noise_schedule_steps = noise_schedule_steps
        if noise_schedule == 'cosine':
            self.noise_schedule = cosine_schedule(noise_schedule_steps)
        else:
            self.noise_schedule = linear_schedule(noise_schedule_steps, *noise_schedule_betas)

    @property
    def token_dim(self):
        return self.backbone.hparams.token_dim  # type: ignore

    def reset_parameters(self):
        for m in self.children():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()  # type: ignore

    @staticmethod
    def _load_backbone(backbone_type, checkpoint_path):
        from src.model.dae import (
            DiscreteAutoencoder, VectorQuantizedAutoencoder, VectorQuantizedGAN
        )
        if backbone_type == "dae":
            return DiscreteAutoencoder.load_from_checkpoint(checkpoint_path)
        elif backbone_type == "vqvae":
            return VectorQuantizedAutoencoder.load_from_checkpoint(checkpoint_path)
        elif backbone_type == "vqgan":
            return VectorQuantizedGAN.load_from_checkpoint(checkpoint_path)
        else:
            raise ValueError(f"Unknown backbone_type: {backbone_type!r}")

    def apply_noise(self, inputs):
        t = torch.randint(0, self.noise_schedule_steps, (len(inputs),), device=inputs.device)
        t = (t + 1).to(torch.float32)

        alpha_bar_t = (self.noise_schedule['alphas_bar'])
        alpha_bar_t = alpha_bar_t[:, *([None] * (len(inputs) - 1))].to(inputs.device)

        noise = torch.randn_like(inputs)
        x_t = torch.sqrt(alpha_bar_t) * inputs + torch.sqrt(1 - alpha_bar_t) * noise

        return x_t, noise, t

    def denoise(self, slots):
        slot_tokens = self.slot_proj(slots)

        x_t = torch.randn(
            (len(slots), *self.resolution, self.token_dim), device=slots.device
        ).flatten(1, 2)

        for t in range(self.noise_schedule_steps + 1, 1, -1):
            beta = self.noise_schedule['betas'][t].to(x_t.device)
            alpha = self.noise_schedule['alphas'][t].to(x_t.device)
            alpha_bar = self.noise_schedule['alphas_bar'][t].to(x_t.device)
            # std = torch.sqrt( 1 - self.noise_schedule['alpha_bar'][t - 1] / (1 - alpha_bar))
            std = torch.sqrt(beta)

            pred_noise = self.denoiser(x_t, slot_tokens, t)
            x_t = (
                1 / torch.sqrt(alpha) * x_t -
                (1 - alpha) / torch.sqrt(alpha * (1 - alpha_bar)) * pred_noise
            )

            if t > 1:
                x_t = x_t + std * torch.randn_like(x_t)

        return x_t


    def forward(self, inputs):
        with torch.no_grad():
            tokens = self.backbone.embed(inputs, reshape='tokenization')

        slots, attn_weigts = self.slot(tokens)
        slot_tokens = self.slot_proj(slots)

        noised_tokens, noise, t = self.apply_noise(tokens)
        pred_noise = self.denoiser(noised_tokens, slot_tokens, t)

        return (pred_noise, noise, t), (slots, attn_weigts), (tokens, noised_tokens)

    def embed(self, inputs):
        tokens = self.backbone.embed(inputs)
        return self.slot(tokens)[0]

    def reconstruction(self, inputs):
        with torch.no_grad():
            tokens = self.backbone.embed(inputs, reshape='tokenization')
        slots, _ = self.slot(tokens)
        slot_tokens = self.slot_proj(slots)
        denoised = self.denoise(slot_tokens)
        return self.backbone.decode(denoised)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        (noise, pred_noise, _), (_, _), (_, _) = self.forward(batch)
        loss = F.mse_loss(noise, pred_noise)
        self.log(
            'train/loss',
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=False,
            rank_zero_only=True
        )
        return loss

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        phase: str = 'val'
    ):
        (noise, pred_noise, _), (slots, _), (tokens, _) = self.forward(batch)
        denoised_tokens = self.denoise(slots)
        recons = self.backbone.decode(denoised_tokens)

        loss = F.mse_loss(noise, pred_noise)
        denoising_loss = F.mse_loss(denoised_tokens, tokens)
        recons_loss = F.mse_loss(recons, batch[0])

        self.log_dict(
            {
                f'{phase}/loss': loss,
                f'{phase}/token_loss': denoising_loss,
                f'{phase}/recons_loss': recons_loss,
            },
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=False,
            rank_zero_only=True
        )

        return loss

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        return self.validation_step(batch, batch_idx, 'test')
