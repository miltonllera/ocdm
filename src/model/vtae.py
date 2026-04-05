from itertools import product
from typing import Callable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.base import BaseModel, TrainingInit
from src.nn.init import linear_init
from src.nn.stochastic import DiagonalGaussian
from src.training.loss import WassersteinMMD
from src.nn.spatial import PositionEmbedding2D
from src.nn.transformer import TransformerEncoder, TransformerDecoder


class VTAE(BaseModel):
    """
    Use a ViT to encode image patches into a single token, which is then use to reconstruct the
    corresponding patch tokens using a Transformer. Analogous to SLATE without segmentation.
    """
    def __init__(
        self,
        backbone_type: Literal["dae", "vqvae", "vqgan"],
        backbone_checkpoint: str,
        training: TrainingInit,
        n_slots: int = 4,
        latent_size: int = 192,
        use_discrete: bool = True,
        lambda1: float = 10.0,
        lambda2: float = 1.0,
        prior_type: Literal["norm", "unif"] = "norm",
        prior_var: float = 1.0,
        kernel: Callable | None = None,
        lambda_schedule: tuple[int, float] | None = None,
        n_head: int = 4,
        num_layers: int = 4,
        ffwd_dim: int | None = None,
        dropout: float = 0.1,
        use_memory_mask: bool = False,
        autoregressive_loss: Literal['xent', 'mse'] = 'xent',
        ar_val_batches: int = 10,

    ):
        super().__init__(training)
        self.save_hyperparameters()

        backbone = VTAE._load_backbone(backbone_type, backbone_checkpoint)
        backbone.requires_grad_(False)
        self.backbone = backbone.eval()
        self.backbone_type = backbone_type

        token_dim = backbone.hparams.token_dim  # type: ignore
        H, W = self.resolution = tuple(backbone.hparams.resolution)  # type: ignore

        # self.pos_emb = PositionEmbedding1D(token_dim, H * W)
        self.pos_emb = PositionEmbedding2D(token_dim, H, W, embed='cardinal')

        self.latent = DiagonalGaussian(token_dim, latent_size)
        self.latent_init = nn.Parameter(torch.empty(1, 1, token_dim))
        self.latent_loss_fn = WassersteinMMD(
            lambda1=lambda1,
            lambda2=lambda2,
            prior_type=prior_type,
            prior_var=prior_var,
            kernel=kernel,
            lambda_schedule=lambda_schedule
        )
        self.latent_proj = nn.Linear(latent_size, token_dim, bias=False)

        self.transformer_encoder = TransformerEncoder(
            d_model=token_dim,
            n_head=n_head,
            num_layers=num_layers,
            ffwd_dim=ffwd_dim,
            dropout=dropout
        )

        max_seqlen = H * W
        self.transformer_decoder = TransformerDecoder(
            max_seqlen=max_seqlen,
            d_model=token_dim,
            n_head=n_head,
            num_layers=num_layers,
            ffwd_dim=ffwd_dim,
            dropout=dropout
        )

        self.use_memory_mask = use_memory_mask
        self.ar_val_batches = ar_val_batches
        self.ar_loss = autoregressive_loss

        if autoregressive_loss == 'xent':
            self.out_proj = nn.Linear(token_dim, backbone.hparams.vocab_size)  # type: ignore
        else:
            self.out_proj = nn.Identity()

        linear_init(self.latent_proj, activation=None)
        nn.init.normal_(self.latent_init)

    @staticmethod
    def _load_backbone(backbone_type, checkpoint_path):
        from src.model.vqae import (
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

    @property
    def vocab_size(self):
        return self.backbone.hparams.vocab_size  # type: ignore

    @property
    def dim(self):
        return self.backbone.hparams.token_dim  # type: ignore

    def forward(self, inputs):
        with torch.no_grad():
            patch_emb, idx = self.backbone.embed(inputs, reshape='tokenization')

        # patch_plus_pos = self.pos_emb(patch_emb)  # N, H * W, E_e
        patch_plus_pos = self.pos_emb(patch_emb.unflatten(1, self.resolution)).flatten(1, 2)

        z, params = self.transformer_encoding(patch_plus_pos)
        z_proj = self.latent_proj(z).unsqueeze(1)

        # NOTE: we predict the true backbone embeddings WITHOUT position information or the index
        # of the feature in the backbone's codebook if using cross_entropy as a loss. Thus the
        # shape of pred_embeddings is either N, H * W, (E_e or C)
        bos_token = self.latent_init.expand(len(inputs), -1, -1)
        tfd_input = torch.cat([bos_token, patch_plus_pos[:, :-1]], dim=1)
        tf_preds = self.out_proj(self.transformer_decoder(tfd_input, z_proj))

        if self.ar_loss == 'xent':
            tf_targets = idx  # index in the backbone codebook
        else:
            tf_targets = patch_emb.detach()  # raw backbone codebook weights

        with torch.no_grad():
            recons = self.backbone.decode(tf_preds, from_idx=self.ar_loss == 'xent')

        return recons, (z, params), (tf_preds, tf_targets)

    def compute_ar_loss(self, pred_tokens, target_tokens):
        B = len(pred_tokens)

        pred_tokens = pred_tokens.flatten(0, 1)
        target_tokens = target_tokens.flatten(0, 1)

        if self.ar_loss == 'mse':
            ar_loss = F.mse_loss(pred_tokens, target_tokens.detach(), reduction='sum') / B
        else:
            ar_loss = F.cross_entropy(pred_tokens, target_tokens, reduction='sum') / B

        return ar_loss

    def transformer_encoding(self, patches):
        tfe_input = torch.cat([self.latent_init.expand(len(patches), -1, -1), patches], dim=1)
        h = self.transformer_encoder(tfe_input)[:, 0]
        z, params = self.latent(h)
        return z, params

    def embed(self, inputs):
        patches = self.backbone.embed(inputs)[0]
        patches = self.pos_emb(patches.unflatten(1, self.resolution)).flatten(1, 2)
        return self.transformer_encoding(patches)[0]

    def reconstruction(self, inputs):
        z = self.embed(inputs)
        return self.autoregressive_recons(z)[0]

    def sample_tokens(self, z):
        H, W = self.resolution

        tf_inputs = self.latent_init.expand(len(z), -1, -1)
        tf_mem = self.latent_proj(z).unsqueeze(1)

        sampled = []
        for pos in product(range(H), range(W)):
            token_pred = self.transformer_decoder(tf_inputs, tf_mem, None)[:, -1:]
            # (B, 1, 1) and (B, 1, E)
            next_emb, next_token = self.backbone.get_quantization(
                self.out_proj(token_pred), from_idx=self.ar_loss == 'xent',
            )
            tf_inputs = torch.cat([tf_inputs, self.pos_emb(next_emb, pos=pos)], dim=1)
            sampled.append(next_token)

        return torch.cat(sampled, dim=1)

    def autoregressive_recons(self, latent):
        with torch.no_grad():
            sampled = self.sample_tokens(latent)
            recons = self.backbone.decode(sampled)
            return recons, sampled

    def _step(self, batch, batch_idx, phase):
        inputs, targets = batch
        recons, (z, z_params), (pred_tokens, target_tokens) = self.forward(inputs)

        ar_loss = self.compute_ar_loss(pred_tokens, target_tokens)
        recons_loss = F.mse_loss(recons, targets, reduction='sum') / len(targets)
        latent_loss = self.latent_loss_fn(z, z_params)

        metrics = {
            f"{phase}/loss": ar_loss + latent_loss,
            f"{phase}/ar_loss": ar_loss,
            f"{phase}/latent_loss": latent_loss,
            f"{phase}/reconstruction_term": recons_loss,
        }

        return recons, (z, z_params), metrics

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ):
        _, _, metrics = self._step(batch, batch_idx, "train")

        self.log_dict(
            metrics,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=False,
            rank_zero_only=True
        )

        return metrics["train/loss"]

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        phase: Literal["val", "test"] = "val"
    ):
        _, (slots, _), metrics = self._step(batch, batch_idx, phase)

        if (phase == "test") or (phase == "val" and batch_idx < self.ar_val_batches):
            ar_recons, _ = self.autoregressive_recons(slots)
            ar_recons_loss = F.mse_loss(
                ar_recons, batch[1], reduction='sum'
            ) / len(batch[1])
            metrics[f"{phase}/ar_sample_loss"] = ar_recons_loss

        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            rank_zero_only=True
        )

        return metrics[f"{phase}/loss"]

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ):
        return self.validation_step(batch, batch_idx, "test")
