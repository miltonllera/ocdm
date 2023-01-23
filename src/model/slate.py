from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn

from src.model.base import BaseModel, OptimizerInit, SchedulerInit
from src.layers.slot import SlotAttention
from src.layers.transformer import TransformerDecoder
from src.layers.stochastic import GumbelSoftmax
from src.layers.token import TokenDict, to_onehot
from src.layers.initialization import linear_init
from src.training.loss import ReconstructionLoss, ImageTokenLoss


class SLATE(BaseModel):
    def __init__(self,
        resolution: Tuple[int, int],
        vocab_size: int,
        dim: int,
        patch_encoder: nn.Sequential,
        patch_decoder: nn.Sequential,
        latent: GumbelSoftmax,
        slot: SlotAttention,
        transformer_decoder: TransformerDecoder,
        loss: ReconstructionLoss,
        optimizer: OptimizerInit,
        scheduler: SchedulerInit,
        scheduler_metric: Optional[str] = None,
        use_memory_mask: bool = False,
        _ar_val_batches: int = 10,
    ):
        super().__init__(optimizer, scheduler, scheduler_metric)
        H, W = resolution

        self.resolution = resolution
        self.patch_encoder = patch_encoder
        self.patch_decoder = patch_decoder
        self.latent = latent
        self.token_dict = TokenDict(vocab_size, dim, H * W)
        self.slot = slot
        self.slot_proj = nn.Linear(dim, dim, bias=False)
        self.transformer_decoder = transformer_decoder
        self.token_logits = nn.Linear(dim, vocab_size, bias=False)
        self.bos_token = nn.Parameter(torch.empty(1, 1, dim))
        self.recons_loss = loss
        self.token_loss = ImageTokenLoss()
        self.use_memory_mask = use_memory_mask
        self._ar_val_batches = _ar_val_batches

        linear_init(self.slot_proj, activation=None)
        linear_init(self.token_logits, activation=None)
        nn.init.normal_(self.bos_token)

    @property
    def vocab_size(self):
        return self.token_dict.vocab_size

    @property
    def dim(self):
        return self.token_dict.embedding_dim

    def forward(self, inputs):
        h = self.patch_encoder(inputs)
        z, logits = self.latent(h)

        recons = self.patch_decoder(z)

        token_idxs = to_onehot(z.flatten(1, 2))

        tokens = self.token_dict(token_idxs)
        tokens = torch.cat(
            [self.bos_token.expand(len(inputs), -1, -1), tokens],
            dim=1
        )

        slot_input = tokens[:, 1:]  # no BOS token in SA input
        tf_input = tokens[:, :-1]  # right shift Transformer input

        slots, attn_weights = self.slot(slot_input)

        proj = self.slot_proj(slots)
        mask = attn_weights.detach() if self.use_memory_mask else None

        token_logits = self.token_logits(
            self.transformer_decoder(tf_input, proj, mask)
        )

        return (recons, (slots, attn_weights), (z, logits),
                (token_logits, token_idxs))

    def _step(self, batch, batch_idx, phase):
        inputs, targets = batch

        recons, slots, zs, tokens = self.forward(inputs)

        recons_loss = self.recons_loss(recons, targets)
        token_loss = self.token_loss(*tokens)
        loss = recons_loss + token_loss

        metrics = {
                f"{phase}/loss": loss,
                f"{phase}/token_xent": token_loss,
                f"{phase}/reconstruction_loss": recons_loss
            }

        return recons, slots, zs, tokens, metrics


    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ):
        recons, _, _, tokens, metrics = self._step(batch, batch_idx, "train")

        self.log_dict(
            metrics,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=False,
            rank_zero_only=True
        )

        # log tau separately so it doesn't appear in the progress bar
        self.log(
            "tau",
            self.latent.tau,
            prog_bar=False,
            on_step=True,
            on_epoch=False
        )

        return metrics["train/loss"]

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        phase: Literal["val", "test"] = "val"
    ):
        _, slots, _, tokens, metrics = self._step(batch, batch_idx, phase)

        if (phase == "test") or (phase == "val" and
            (batch_idx < self._ar_val_batches)):
            target_tokens, target_images = tokens[1], batch[1]

            ar_recons, sampled_tokens = self.autoregressive_recons(slots)

            ar_recons_loss = self.recons_loss(ar_recons, target_images)
            ar_token_xent = self.token_loss(sampled_tokens, target_tokens)

            metrics[f"{phase}/autoregressive_recons"] = ar_recons_loss
            metrics[f"{phase}/autoregressive_token_xent"] = ar_token_xent

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
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ):
        return self.validation_step(batch, batch_idx, "test")

    def embed(self, inputs):
        h = self.patch_encoder(inputs)
        z, _ = self.latent(h)

        token_idxs = to_onehot(z.flatten(1, 2))
        tokens = self.token_dict(token_idxs)

        return self.slot(tokens)

    def autoregressive_recons(self, slots):
        sampled_discrete = self.sample_tokens(
            *slots).to(dtype=torch.float32)

        recons = self.patch_decoder(
            sampled_discrete.unflatten(1, self.resolution)
        )

        return recons, sampled_discrete

    def sample_tokens(self, slots, attn_weights=None):
        """
        Sample tokens autoregressively using the Transformer decoder.
        """
        H, W = self.resolution
        slot_proj = self.slot_proj(slots)

        if attn_weights is not None and self.use_memory_mask:
            mask = attn_weights.detach()
        else:
            mask = None

        sampled_discrete = []
        token_inputs = self.bos_token.expand(len(slots), -1, -1)

        for i in range(H * W):
            u = self.transformer_decoder(token_inputs, slot_proj, mask)[:, -1:]
            new_token_idx = to_onehot(self.token_logits(u))

            sampled_discrete.append(new_token_idx)

            new_token = self.token_dict(new_token_idx, start_pos=i)
            token_inputs = torch.cat([token_inputs, new_token], dim=1)

        return torch.cat(sampled_discrete, dim=1)
