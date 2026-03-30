from itertools import product
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.base import BaseModel, TrainingInit
from src.nn.init import linear_init
from src.nn.slot import SlotAttention
from src.nn.spatial import PositionEmbedding2D
from src.nn.transformer import TransformerDecoder


class SLATE(BaseModel):
    """
    SLATE with a pretrained frozen tokenizer backbone (DiscreteAutoencoder, VQ-VAE or VQ-GAN).
    Trains only the slot attention + transformer decoder with MeanSquaredError on token embeddings.
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
        n_head: int = 4,
        num_layers: int = 4,
        ffwd_dim: int | None = None,
        dropout: float = 0.1,
        use_memory_mask: bool = False,
        _ar_val_batches: int = 10,
    ):
        super().__init__(training)
        self.save_hyperparameters()

        backbone = SLATE._load_backbone(backbone_type, backbone_checkpoint)
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

        max_seqlen = H * W
        self.transformer_decoder = TransformerDecoder(
            max_seqlen=max_seqlen,
            d_model=token_dim,
            n_head=n_head,
            num_layers=num_layers,
            ffwd_dim=ffwd_dim,
            dropout=dropout
        )

        self.slot_out_proj = nn.Linear(slot_size, token_dim, bias=False)
        self.bos_token = nn.Parameter(torch.empty(1, 1, token_dim))

        self.use_memory_mask = use_memory_mask
        self._ar_val_batches = _ar_val_batches

        linear_init(self.slot_out_proj, activation=None)
        nn.init.normal_(self.bos_token)

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

    @property
    def vocab_size(self):
        return self.backbone.hparams.vocab_size  # type: ignore

    @property
    def dim(self):
        return self.backbone.hparams.token_dim  # type: ignore

    def _nearest_token(self, pred_emb):
        B, S, _ = pred_emb.shape
        if self.backbone_type == "discrete":
            raise NotImplementedError()
        else:
            z_q, idx, _ = self.backbone.feature_codebook(pred_emb.flatten(0, -2))  # type: ignore
            return idx.unflatten(0, (B, S)), z_q.unflatten(0, (B, S))

    def forward(self, inputs):
        with torch.no_grad():
            tokens = self.backbone.embed(inputs, reshape='tokenization')

        tokens = torch.cat([self.bos_token.expand(len(inputs), -1, -1), tokens], dim=1)

        slot_input = tokens[:, 1:]
        tf_input = tokens[:, :-1]

        slots, attn_weights = self.slot(slot_input)
        mask = attn_weights.detach() if self.use_memory_mask else None
        slot_tokens = self.slot_out_proj(slots)

        pred_embeddings = self.transformer_decoder(tf_input, slot_tokens, mask)

        with torch.no_grad():
            recons = self.backbone.decode(pred_embeddings)

        return recons, (slots, attn_weights), (pred_embeddings, tokens[:, 1:])

    def _step(self, batch, batch_idx, phase):
        inputs, targets = batch
        recons, slots, (pred_embeddings, target_embeddings) = self.forward(inputs)
        emb_loss = F.mse_loss(pred_embeddings, target_embeddings.detach())
        recons_loss = F.mse_loss(recons, targets, reduction='sum') / len(targets)

        metrics = {
            f"{phase}/loss": emb_loss,
            f"{phase}/reconstruction_term": recons_loss,
        }

        return recons, slots, metrics

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

        if (phase == "test") or (phase == "val" and batch_idx < self._ar_val_batches):
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

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.embed(inputs)

    def reconstruction(self, inputs):
        slots = self.embed(inputs)
        return self.autoregressive_recons(slots)[0]

    def embed(self, inputs):
        tokens = self.backbone.embed(inputs)
        return self.slot(tokens)[0]

    def autoregressive_recons(self, slots):
        with torch.no_grad():
            sampled = self.sample_tokens(slots).to(dtype=torch.float32)
            recons = self.backbone.decode(sampled)
            return recons, sampled

    def sample_tokens(self, slots, use_codebook_emb=True):
        H, W = self.resolution
        slot_proj = self.slot_out_proj(slots)

        sampled = []
        token_inputs = self.bos_token.expand(len(slots), -1, -1)

        for pos in product(range(H), range(W)):
            pred_emb = self.transformer_decoder(token_inputs, slot_proj, None)[:, -1:]
            if use_codebook_emb:
                _, new_codebook_emb = self._nearest_token(pred_emb)
            else:
                new_codebook_emb = pred_emb
            sampled.append(new_codebook_emb)
            new_token = self.pos_emb(new_codebook_emb, pos=pos)
            token_inputs = torch.cat([token_inputs, new_token], dim=1)

        return torch.cat(sampled, dim=1)
