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
    Trains only the slot attention + transformer decoder modules with either MSE or CrossEntropy
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
        autoregressive_loss: Literal['xent', 'mse'] = 'xent',
        ar_val_batches: int = 10,

    ):
        super().__init__(training)
        self.save_hyperparameters()

        backbone = SLATE._load_backbone(backbone_type, backbone_checkpoint)
        backbone.requires_grad_(False)
        self.backbone = backbone.eval()
        self.backbone_type = backbone_type

        token_dim = backbone.hparams.token_dim  # type: ignore
        H, W = self.resolution = tuple(backbone.hparams.resolution)  # type: ignore

        # self.pos_emb = PositionEmbedding1D(token_dim, H * W)
        self.pos_emb = PositionEmbedding2D(token_dim, H, W, embed='cardinal')

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
        self.ar_val_batches = ar_val_batches
        self.ar_loss = autoregressive_loss

        if autoregressive_loss == 'xent':
            self.out_proj = nn.Linear(token_dim, backbone.hparams.vocab_size)  # type: ignore
        else:
            self.out_proj = nn.Identity()

        linear_init(self.slot_out_proj, activation=None)
        nn.init.normal_(self.bos_token)

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
            patches, idx = self.backbone.embed(inputs, reshape='tokenization')

        # N, H * W, E_e
        # patch_plus_pos = self.pos_emb(patch_emb)  # N, H * W, E_e
        patch_plus_pos = self.pos_emb(patches.unflatten(1, self.resolution)).flatten(1, 2)

        slots, attn_weights = self.slot(patch_plus_pos)  # N, S, E_s
        mask = attn_weights.detach() if self.use_memory_mask else None

        tf_input = torch.cat(  # N, H * W, E_e
            [self.bos_token.expand(len(inputs), -1, -1), patch_plus_pos[:, :-1]],
            dim=1
        )
        tf_mem = self.slot_out_proj(slots)  # N, S, E_e

        # NOTE: we predict the true backbone embeddings WITHOUT position information if using MSE,
        # or the index logits of the feature in the backbone's codebook if using cross_entropy.
        # Thus the shape of pred_embeddings is either N, H * W, (E_e or C)
        tf_preds = self.out_proj(self.transformer_decoder(tf_input, tf_mem, mask))

        if self.ar_loss == 'xent':
            tf_targets = idx  # index in the backbone's codebook
        else:
            tf_targets = patches.detach()  # raw backbone codebook weight

        with torch.no_grad():
            recons = self.backbone.decode(tf_preds, from_idx=self.ar_loss == 'xent')

        return recons, (slots, attn_weights), (tf_preds, tf_targets)

    def compute_ar_loss(self, pred_tokens, target_tokens):
        B = len(pred_tokens)

        pred_tokens = pred_tokens.flatten(0, 1)
        target_tokens = target_tokens.flatten(0, 1)

        if self.ar_loss == 'mse':
            ar_loss = F.mse_loss(pred_tokens, target_tokens.detach(), reduction='sum') / B
        else:
            ar_loss = F.cross_entropy(pred_tokens, target_tokens, reduction='sum') / B

        return ar_loss

    def embed(self, inputs):
        patches = self.backbone.embed(inputs)[0]
        patches = self.pos_emb(patches.unflatten(1, self.resolution)).flatten(1, 2)
        return self.slot(patches)[0]

    def reconstruction(self, inputs):
        slots = self.embed(inputs)
        return self.autoregressive_recons(slots)[0]

    def sample_tokens(self, slots):
        H, W = self.resolution

        tf_mem = self.slot_out_proj(slots)
        tf_inputs = self.bos_token.expand(len(slots), -1, -1)

        sampled = []
        for pos in product(range(H), range(W)):
            token_pred = self.transformer_decoder(tf_inputs, tf_mem, None)[:, -1:]
            # (B, 1) and (B, 1, E), with types torch.long and torch.float
            next_emb, next_token = self.backbone.get_quantization(
                self.out_proj(token_pred), from_idx=self.ar_loss == 'xent',
            )
            tf_inputs = torch.cat([tf_inputs, self.pos_emb(next_emb, pos=pos)], dim=1)
            sampled.append(next_token)

        return torch.cat(sampled, dim=1)

    def autoregressive_recons(self, slots):
        with torch.no_grad():
            sampled = self.sample_tokens(slots)
            recons = self.backbone.decode(sampled, from_idx=True)
            return recons, sampled

    def _step(self, batch, batch_idx, phase):
        inputs, targets = batch
        recons, (slots, attn), (pred_tokens, target_tokens) = self.forward(inputs)

        ar_loss = self.compute_ar_loss(pred_tokens, target_tokens)
        recons_loss = F.mse_loss(recons, targets, reduction='sum') / len(targets)

        # treat slot assignment for each patch as prob
        # attn_log_probs = F.softmax(attn, dim=2)
        # # compute entropy for each patch
        # patch_attn_entropy = -(attn_log_probs * torch.log(attn_log_probs + 1e-8)).sum(dim=2)
        # attn_entropy_loss = patch_attn_entropy.sum() / len(inputs)

        # loss = ar_loss + attn_entropy_loss

        loss = ar_loss

        metrics = {
            f"{phase}/loss": loss,
            f"{phase}/ar_loss": ar_loss,
            # f"{phase}/attn_entropy_loss": attn_entropy_loss,
            f"{phase}/reconstruction_term": recons_loss
        }

        return recons, (slots, attn), metrics

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
