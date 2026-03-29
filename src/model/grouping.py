from itertools import product
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.base import BaseModel, TrainingInit
from src.nn.init import linear_init
from src.nn.slot import SlotAttention, FigureGroundSegmentation
from src.nn.stochastic import GumbelSoftmax, DiagonalGaussian
from src.nn.token import SpatialTokenDict, to_onehot
from src.nn.transformer import TransformerDecoder
from src.nn.utils.parsing import create_sequential
from src.training.loss import ImageTokenLoss


#----------------------------------------- Slot variants -----------------------------------------

class SlotAutoencoder(BaseModel):
    def __init__(
        self,
        input_size: tuple[int, int, int],
        encoder_config: list,
        decoder_config: list,
        training: TrainingInit,
        n_slots: int = 4,
        slot_size: int = 64,
        n_iter: int = 3,
        slot_channels: int = 1,
        slot_hidden_size: int = 128,
        approx_implicit_grad: bool = True,
    ):
        super().__init__(training)
        self.save_hyperparameters()

        # Build encoder from config
        self.encoder = create_sequential(input_size, encoder_config)

        # Get encoder output size for slot attention input
        with torch.no_grad():
            if isinstance(input_size, int):
                dummy_input = torch.zeros(1, input_size)
            else:
                dummy_input = torch.zeros(1, *input_size)
            encoder_output = self.encoder(dummy_input)
            encoder_output_size = encoder_output.shape[-1]

        # Build slot attention with explicit parameters
        self.slot_attention = SlotAttention(
            input_size=encoder_output_size,
            n_slots=n_slots,
            slot_size=slot_size,
            n_iter=n_iter,
            slot_channels=slot_channels,
            hidden_size=slot_hidden_size,
            approx_implicit_grad=approx_implicit_grad
        )

        # Build decoder from config
        self.decoder = create_sequential(slot_size, decoder_config)

        # Fixed MSE reconstruction loss
        self.recons_loss = nn.MSELoss(reduction='sum')

    def decode(
        self, slots: torch.Tensor, attention_weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode slots into reconstructions and masks."""
        batch_size, n_slots = slots.size()[:2]

        # Batchify reconstruction - flatten slots for processing
        slots_flat = slots.flatten(end_dim=1)

        # Pass through decoder network to get RGBA output
        rgba = self.decoder(slots_flat)

        # Reshape back to (batch_size, n_slots, channels, height, width)
        rgba = rgba.unflatten(0, (batch_size, n_slots))

        # Split into reconstructions and mask logits (last channel is mask)
        slot_recons, slot_mask_logits = torch.tensor_split(rgba, indices=[-1], dim=2)

        # Apply softmax to mask logits across slots dimension
        slot_masks = torch.softmax(slot_mask_logits, dim=1)

        # Combine slot reconstructions weighted by masks
        recons = (slot_masks * slot_recons).sum(dim=1)

        return recons, slot_masks

    def forward(
            self,
            inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(inputs)
        slots, attention_weights = self.slot_attention(h)
        recons, decoder_masks = self.decode(slots, attention_weights)
        return recons, slots, decoder_masks

    def _step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        phase: Literal["train", "val", "test"]
    ) -> torch.Tensor:
        inputs, targets = batch
        targets = 2 * targets - 1  # re-scale targets to -1; 1

        recons, _, _ = self.forward(inputs)

        loss = self.recons_loss(recons, targets) / len(targets)
        is_train = phase == "train"
        self.log(
            f"{phase}/loss",
            loss,
            on_epoch=not is_train,
            on_step=is_train,
            prog_bar=is_train,
            sync_dist=not is_train,
            rank_zero_only=True
        )

        # if is_train:
        #     loss.backward()
        #     for name, param in self.named_parameters():
        #         if param.grad is None:
        #             print(name)
        #     exit()

        return loss

    def predict(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self(inputs)

    def embed(self, inputs: torch.Tensor) -> torch.Tensor:
        """Extract slot representations from inputs."""
        h = self.encoder(inputs)
        slots, _ = self.slot_attention(h)
        return slots

    def reconstruction(self, inputs):
        output = (self.predict(inputs)[0] + 1) / 2
        return output.clip(0, 1)

    # def on_validation_epoch_end(self):
    #     # Sample a few validation images
    #     val_loader = self.val_dataloader()
    #     x, _ = next(iter(val_loader))
    #     x = x.to(self.device)

    #     with torch.no_grad():
    #         x_hat, _, _ = self(x)

    #     n = 10
    #     x = x[:n]
    #     x_hat = x[:n]

    #     # Log to W&B as an image grid
    #     comparison = torch.cat([x, x_hat])
    #     grid = make_grid(comparison, nrow=n, normalize=True)

    #     # Convert grid to W&B Image

    #     wandb_img = wandb.Image(grid, caption=f"Reconstructions @ epoch {self.current_epoch}")
    #     self.logger.experiment.log({"reconstructions": wandb_img, "epoch": self.current_epoch})


class FigureGroundAutoencoder(BaseModel):
    def __init__(
        self,
        input_size: tuple[int, int, int],
        encoder_config: list,
        decoder_config: list,
        training: TrainingInit,
        slot_size: int = 64,
        n_iter: int = 3,
        slot_channels: int = 1,
        slot_hidden_size: int = 128,
        approx_implicit_grad: bool = False,
    ):
        super().__init__(training)
        self.save_hyperparameters()

        # Build encoder from config
        self.encoder = create_sequential(input_size, encoder_config)

        # Get encoder output size for figure-ground segmentation input
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_size)
            encoder_output = self.encoder(dummy_input)
            encoder_output_size = encoder_output.shape[-1]

        # Build figure-ground segmentation with explicit parameters
        self.fig_rep = FigureGroundSegmentation(
            input_size=encoder_output_size,
            latent_size=slot_size,
            n_iter=n_iter,
            n_channels=slot_channels,
            hidden_size=slot_hidden_size,
            approx_implicit_grad=approx_implicit_grad
        )

        # Build decoder from config
        self.decoder = create_sequential(slot_size, decoder_config)

        # Fixed MSE reconstruction loss
        self.recons_loss = nn.MSELoss(reduction='sum')

    def decode(
        self,
        fig_reps: torch.Tensor,
        attention_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode figure representations into reconstructions and masks."""
        # Remove the slot dimension (fig_reps is (batch_size, 1, latent_size))
        fig_reps_flat = fig_reps.squeeze(1)

        # Pass through decoder network to get RGBA output
        rgba = self.decoder(fig_reps_flat)

        # Split into reconstructions and mask logits (last channel is mask)
        fig_recons, fig_mask_logits = torch.tensor_split(rgba, indices=[-1], dim=1)

        # Apply sigmoid to mask logits (binary mask for figure vs background)
        fig_masks = torch.sigmoid(fig_mask_logits)

        # Combine figure reconstruction weighted by mask
        recons = fig_masks * fig_recons

        return recons, fig_masks

    def forward(self, inputs: torch.Tensor):
        h = self.encoder(inputs)
        slots, attention_weights = self.fig_rep(h)
        recons, decoder_masks = self.decode(slots, attention_weights)
        return recons, slots, decoder_masks

    def _step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        phase: Literal["train", "val", "test"]
    ):
        inputs, targets = batch
        # targets = 2 * targets - 1  # re-scale targets to -1; 1

        recons, _, _ = self.forward(inputs)

        loss = self.recons_loss(recons, targets) / len(targets)

        is_train = phase == "train"
        self.log(
            f"{phase}/loss",
            loss,
            on_epoch=not is_train,
            on_step=is_train,
            prog_bar=is_train,
            sync_dist=not is_train,
            rank_zero_only=True
        )

        return loss

    def predict(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self(inputs)

    def embed(self, inputs):
        h = self.encoder(inputs)
        slots, _ = self.fig_rep(h)
        return slots

    def reconstruction(self, inputs):
        # output = (self.predict(inputs)[0] + 1) / 2
        output = self.predict(inputs)[0]
        return output.clip(0, 1)


#---------------------------------------- SLATE variants -----------------------------------------

class SLATE(BaseModel):
    def __init__(
        self,
        input_size: tuple[int, int, int],
        patch_encoder_config: list,
        patch_decoder_config: list,
        training: TrainingInit,
        resolution: tuple[int, int] = (8, 8),
        vocab_size: int = 4096,
        token_dim: int = 192,
        # GumbelSoftmax parameters
        tau: float = 1.0,
        tau_start: float | None = None,
        tau_steps: float | None = None,
        # SlotAttention parameters
        n_slots: int = 4,
        slot_size: int = 192,
        slot_n_iter: int = 3,
        slot_channels: int = 1,
        slot_hidden_size: int = 128,
        slot_approx_implicit_grad: bool = True,
        # TransformerDecoder parameters
        n_head: int = 4,
        num_layers: int = 4,
        ffwd_dim: int | None = None,
        dropout: float = 0.1,
        # Other parameters
        use_memory_mask: bool = False,
        _ar_val_batches: int = 10,
    ):
        super().__init__(training)
        self.save_hyperparameters()
        H, W = self.resolution = resolution

        # Build patch encoder from config
        self.patch_encoder = create_sequential(input_size, patch_encoder_config)

        # Get patch encoder output size for latent layer input
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_size)
            patch_output = self.patch_encoder(dummy_input)
            assert patch_output.shape[1:-1] == (H, W)
            patch_output_size = patch_output.shape[-1]

        # Build GumbelSoftmax latent layer
        self.latent = GumbelSoftmax(
            input_size=patch_output_size,
            n_cat=vocab_size,
            tau=tau,
            tau_start=tau_start,
            tau_steps=tau_steps
        )

        # Build patch decoder from config
        decoder_input_size = (H, W, vocab_size)
        self.patch_decoder = create_sequential(decoder_input_size, patch_decoder_config)

        # Build token dictionary and slot attention
        self.token_dict = SpatialTokenDict(vocab_size, token_dim, H, W)

        self.slot = SlotAttention(
            input_size=token_dim,
            n_slots=n_slots,
            slot_size=slot_size,
            n_iter=slot_n_iter,
            slot_channels=slot_channels,
            hidden_size=slot_hidden_size,
            approx_implicit_grad=slot_approx_implicit_grad
        )

        # Build transformer decoder
        max_seqlen = H * W
        self.transformer_decoder = TransformerDecoder(
            max_seqlen=max_seqlen,
            d_model=token_dim,
            n_head=n_head,
            num_layers=num_layers,
            ffwd_dim=ffwd_dim,
            dropout=dropout
        )

        # Projection and output layers
        self.slot_out_proj = nn.Linear(slot_size, token_dim, bias=False)
        self.token_logits = nn.Linear(token_dim, vocab_size, bias=False)
        self.bos_token = nn.Parameter(torch.empty(1, 1, token_dim))

        # Fixed MSE reconstruction loss
        self.recons_loss = nn.MSELoss(reduction='sum')
        self.token_loss = ImageTokenLoss()

        self.use_memory_mask = use_memory_mask
        self._ar_val_batches = _ar_val_batches

        linear_init(self.slot_out_proj, activation=None)
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

        token_idxs = to_onehot(z)
        tokens = self.token_dict(token_idxs).flatten(1, 2)
        tokens = torch.cat(
            [self.bos_token.expand(len(inputs), -1, -1), tokens],
            dim=1
        )

        slot_input = tokens[:, 1:]  # no BOS token in SA input
        tf_input = tokens[:, :-1]  # right shift Transformer input

        slots, attn_weights = self.slot(slot_input)

        proj = self.slot_out_proj(slots)
        mask = attn_weights.detach() if self.use_memory_mask else None

        token_logits = self.token_logits(
            self.transformer_decoder(tf_input, proj, mask)
        )

        return (
            recons,
            (slots, attn_weights),
            (z, logits),
            (token_logits, token_idxs.flatten(1, 2))
        )

    def _step(self, batch, batch_idx, phase):
        inputs, targets = batch
        targets = 2 * targets - 1  # re-scale targets to -1; 1

        recons, slots, zs, tokens = self.forward(inputs)

        recons_loss = self.recons_loss(recons, targets) / len(targets)
        token_loss = self.token_loss(*tokens)
        loss = recons_loss + token_loss

        metrics = {
                f"{phase}/loss": loss,
                f"{phase}/token_xent": token_loss,
                f"{phase}/reconstruction_term": recons_loss
            }

        return recons, slots, zs, tokens, metrics

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ):
        _, _, _, _, metrics = self._step(batch, batch_idx, "train")

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
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        phase: Literal["val", "test"] = "val"
    ):
        _, slots, _, tokens, metrics = self._step(batch, batch_idx, phase)

        if (phase == "test") or (phase == "val" and (batch_idx < self._ar_val_batches)):
            target_tokens, target_images = tokens[1], batch[1]

            ar_recons, sampled_tokens = self.autoregressive_recons(slots)

            ar_recons_loss = self.recons_loss(ar_recons, target_images) / len(target_images)
            ar_token_xent = self.token_loss(sampled_tokens, target_tokens)

            metrics[f"{phase}/reconstruction_term"] = ar_recons_loss
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
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ):
        return self.validation_step(batch, batch_idx, "test")

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.embed(inputs)

    def reconstruction(self, inputs):
        slots = self.embed(inputs)
        outputs = (self.autoregressive_recons(slots)[0] + 1) / 2
        return outputs.clip(0, 1)

    def embed(self, inputs):
        h = self.patch_encoder(inputs)
        z, _ = self.latent(h)

        token_idxs = to_onehot(z)
        tokens = self.token_dict(token_idxs).flatten(1, 2)

        return self.slot(tokens)

    def autoregressive_recons(self, slots):
        with torch.no_grad():
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
        slot_proj = self.slot_out_proj(slots)

        if attn_weights is not None and self.use_memory_mask:
            mask = attn_weights.detach()
        else:
            mask = None

        sampled_discrete = []
        token_inputs = self.bos_token.expand(len(slots), -1, -1)

        for pos in product(range(H), range(W)):
            u = self.transformer_decoder(token_inputs, slot_proj, mask)[:, -1:]
            new_token_idx = to_onehot(self.token_logits(u))

            sampled_discrete.append(new_token_idx)

            new_token = self.token_dict(new_token_idx, pos=pos)
            token_inputs = torch.cat([token_inputs, new_token], dim=1)

        return torch.cat(sampled_discrete, dim=1)


class FigureGroundSLATE(BaseModel):
    def __init__(
        self,
        input_size: tuple[int, int, int],
        patch_encoder_config: list,
        patch_decoder_config: list,
        training: TrainingInit,
        resolution: tuple[int, int] = (8, 8),
        vocab_size: int = 4096,
        token_dim: int = 192,
        # GumbelSoftmax parameters
        tau: float = 1.0,
        tau_start: float | None = None,
        tau_steps: float | None = None,
        # SlotAttention parameters
        slot_size: int = 192,
        slot_n_iter: int = 3,
        slot_channels: int = 1,
        slot_hidden_size: int = 128,
        slot_approx_implicit_grad: bool = True,
        # TransformerDecoder parameters
        n_head: int = 4,
        num_layers: int = 4,
        ffwd_dim: int | None = None,
        dropout: float = 0.1,
        # Other parameters
        use_memory_mask: bool = False,
        _ar_val_batches: int = 10,
    ):
        super().__init__(training)
        self.save_hyperparameters()
        H, W = self.resolution = resolution

        # Build patch encoder from config
        self.patch_encoder = create_sequential(input_size, patch_encoder_config)

        # Get patch encoder output size for latent layer input
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_size)
            patch_output = self.patch_encoder(dummy_input)
            assert patch_output.shape[1:-1] == (H, W)
            patch_output_size = patch_output.shape[-1]

        # Build GumbelSoftmax latent layer
        self.latent = GumbelSoftmax(
            input_size=patch_output_size,
            n_cat=vocab_size,
            tau=tau,
            tau_start=tau_start,
            tau_steps=tau_steps
        )

        # Build patch decoder from config
        decoder_input_size = (H, W, vocab_size)
        self.patch_decoder = create_sequential(decoder_input_size, patch_decoder_config)

        # Build token dictionary and slot attention
        self.token_dict = SpatialTokenDict(vocab_size, token_dim, H, W)

        self.slot = FigureGroundSegmentation(
            input_size=token_dim,
            latent_size=slot_size,
            n_iter=slot_n_iter,
            n_channels=slot_channels,
            hidden_size=slot_hidden_size,
            approx_implicit_grad=slot_approx_implicit_grad
        )
        self.background_slot = nn.Parameter(torch.empty(slot_size))
        nn.init.uniform_(self.background_slot)

        # Build transformer decoder
        max_seqlen = H * W
        self.transformer_decoder = TransformerDecoder(
            max_seqlen=max_seqlen,
            d_model=token_dim,
            n_head=n_head,
            num_layers=num_layers,
            ffwd_dim=ffwd_dim,
            dropout=dropout
        )

        # Projection and output layers
        self.slot_out_proj = nn.Linear(slot_size, token_dim, bias=False)
        self.token_logits = nn.Linear(token_dim, vocab_size, bias=False)
        self.bos_token = nn.Parameter(torch.empty(1, 1, token_dim))

        # Fixed MSE reconstruction loss
        self.recons_loss = nn.MSELoss(reduction='sum')
        self.token_loss = ImageTokenLoss()

        self.use_memory_mask = use_memory_mask
        self._ar_val_batches = _ar_val_batches

        linear_init(self.slot_out_proj, activation=None)
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

        token_idxs = to_onehot(z)
        tokens = self.token_dict(token_idxs).flatten(1, 2)
        tokens = torch.cat(
            [self.bos_token.expand(len(inputs), -1, -1), tokens],
            dim=1
        )

        slot_input = tokens[:, 1:]  # no BOS token in SA input
        tf_input = tokens[:, :-1]  # right shift Transformer input

        slots, attn_weights = self.slot(slot_input)
        background_slot = torch.tile(self.background_slot[None, None], (len(inputs), 1, 1))
        slots = torch.cat([slots, background_slot], dim=1)

        proj = self.slot_out_proj(slots)
        mask = attn_weights.detach() if self.use_memory_mask else None

        token_logits = self.token_logits(
            self.transformer_decoder(tf_input, proj, mask)
        )

        return (
            recons,
            (slots, attn_weights),
            (z, logits),
            (token_logits, token_idxs.flatten(1, 2))
        )

    def _step(self, batch, batch_idx, phase):
        inputs, targets = batch
        targets = 2 * targets - 1

        recons, slots, zs, tokens = self.forward(inputs)

        recons_loss = self.recons_loss(recons, targets) / len(targets)
        token_loss = self.token_loss(*tokens)
        loss = recons_loss + token_loss

        metrics = {
                f"{phase}/loss": loss,
                f"{phase}/token_xent": token_loss,
                f"{phase}/reconstruction_term": recons_loss
            }

        return recons, slots, zs, tokens, metrics

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
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
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        phase: Literal["val", "test"] = "val"
    ):
        _, slots, _, tokens, metrics = self._step(batch, batch_idx, phase)

        if (phase == "test") or (phase == "val" and (batch_idx < self._ar_val_batches)):
            target_tokens, target_images = tokens[1], batch[1]

            ar_recons, sampled_tokens = self.autoregressive_recons(slots)

            ar_recons_loss = self.recons_loss(ar_recons, target_images) / len(target_images)
            ar_token_xent = self.token_loss(sampled_tokens, target_tokens)

            metrics[f"{phase}/reconstruction_term"] = ar_recons_loss
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
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ):
        return self.validation_step(batch, batch_idx, "test")

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.embed(inputs)

    def reconstruction(self, inputs):
        slots, _ = self.embed(inputs)
        background_slot = torch.tile(self.background_slot[None, None], (len(inputs), 1, 1))
        slots = torch.cat([slots, background_slot], dim=1)
        outputs = (self.autoregressive_recons((slots, None))[0] + 1) / 2
        return outputs.clip(0, 1)

    def embed(self, inputs):
        h = self.patch_encoder(inputs)
        z, _ = self.latent(h)

        token_idxs = to_onehot(z)
        tokens = self.token_dict(token_idxs).flatten(1, 2)

        return self.slot(tokens)

    def autoregressive_recons(self, slots):
        with torch.no_grad():
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
        slot_proj = self.slot_out_proj(slots)

        if attn_weights is not None and self.use_memory_mask:
            mask = attn_weights.detach()
        else:
            mask = None

        sampled_discrete = []
        token_inputs = self.bos_token.expand(len(slots), -1, -1)

        for pos in product(range(H), range(W)):
            u = self.transformer_decoder(token_inputs, slot_proj, mask)[:, -1:]
            new_token_idx = to_onehot(self.token_logits(u))

            sampled_discrete.append(new_token_idx)

            new_token = self.token_dict(new_token_idx, pos=pos)
            token_inputs = torch.cat([token_inputs, new_token], dim=1)

        return torch.cat(sampled_discrete, dim=1)



class BackboneSLATE(BaseModel):
    """SLATE with a pretrained frozen tokenizer backbone (DiscreteAutoencoder, VQ-VAE or VQ-GAN).

    Trains only the slot attention + transformer decoder with cross-entropy on token IDs.
    """

    def __init__(
        self,
        backbone_type: Literal["discrete", "vqvae", "vqgan"],
        backbone_checkpoint: str,
        training: TrainingInit,
        token_dim: int = 192,
        # SlotAttention parameters
        n_slots: int = 4,
        slot_size: int = 192,
        slot_n_iter: int = 3,
        slot_channels: int = 1,
        slot_hidden_size: int = 128,
        slot_approx_implicit_grad: bool = True,
        # TransformerDecoder parameters
        n_head: int = 4,
        num_layers: int = 4,
        ffwd_dim: int | None = None,
        dropout: float = 0.1,
        use_memory_mask: bool = False,
        _ar_val_batches: int = 10,
    ):
        super().__init__(training)
        self.save_hyperparameters()

        backbone = BackboneSLATE._load_backbone(backbone_type, backbone_checkpoint)
        backbone.requires_grad_(False)
        self.backbone = backbone
        self.backbone_type = backbone_type

        vocab_size = backbone.hparams.vocab_size
        H, W = self.resolution = tuple(backbone.hparams.resolution)

        self.token_dict = SpatialTokenDict(vocab_size, token_dim, H, W)

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
        self.token_logits = nn.Linear(token_dim, vocab_size, bias=False)
        self.bos_token = nn.Parameter(torch.empty(1, 1, token_dim))

        self.token_loss = ImageTokenLoss()
        self.use_memory_mask = use_memory_mask
        self._ar_val_batches = _ar_val_batches

        linear_init(self.slot_out_proj, activation=None)
        linear_init(self.token_logits, activation=None)
        nn.init.normal_(self.bos_token)

    @staticmethod
    def _load_backbone(backbone_type, checkpoint_path):
        from src.model.autoencoder import (
            DiscreteAutoencoder, VectorQuantizedAutoencoder, VectorQuantizedGAN
        )
        if backbone_type == "discrete":
            return DiscreteAutoencoder.load_from_checkpoint(checkpoint_path)
        elif backbone_type == "vqvae":
            return VectorQuantizedAutoencoder.load_from_checkpoint(checkpoint_path)
        elif backbone_type == "vqgan":
            return VectorQuantizedGAN.load_from_checkpoint(checkpoint_path)
        else:
            raise ValueError(f"Unknown backbone_type: {backbone_type!r}")

    @property
    def vocab_size(self):
        return self.token_dict.vocab_size

    @property
    def dim(self):
        return self.token_dict.embedding_dim

    def _get_token_idxs(self, token_output):
        """Convert second output of backbone.forward to one-hot [B, H, W, vocab_size]."""
        if self.backbone_type == "discrete":
            return to_onehot(token_output)
        else:
            return F.one_hot(token_output, self.vocab_size).float()

    def _decode_tokens(self, token_idxs):
        """Decode one-hot token indices [B, H*W, vocab_size] to pixel space."""
        B = len(token_idxs)
        H, W = self.resolution
        if self.backbone_type == "discrete":
            features = token_idxs.flatten(0, 1) @ self.backbone.feature_dict
            return self.backbone.patch_decoder(
                features.unflatten(0, (B, H, W)).permute(0, 3, 1, 2)
            )
        else:
            print(token_idxs.shape)
            exit()
            return self.backbone.decode(token_idxs.argmax(-1))

    def forward(self, inputs):
        with torch.no_grad():
            _, token_idxs, *_ = self.backbone(inputs)
            print(token_idxs.shape)
            token_targets = self._get_token_idxs(token_idxs)

        print(token_targets.shape)
        exit()

        tokens = self.token_dict(token_targets).flatten(1, 2)
        tokens = torch.cat(
            [self.bos_token.expand(len(inputs), -1, -1), tokens],
            dim=1
        )

        slot_input = tokens[:, 1:]
        tf_input = tokens[:, :-1]

        slots, attn_weights = self.slot(slot_input)
        proj = self.slot_out_proj(slots)
        mask = attn_weights.detach() if self.use_memory_mask else None

        token_logits = self.token_logits(
            self.transformer_decoder(tf_input, proj, mask)
        )

        with torch.no_grad():
            pred_onehot = F.one_hot(token_logits.argmax(-1), self.vocab_size).float()
            recons = self._decode_tokens(pred_onehot)

        return recons, (slots, attn_weights), (token_logits, token_targets.flatten(1, 2))

    def _step(self, batch, batch_idx, phase):
        inputs, targets = batch
        recons, slots, tokens = self.forward(inputs)
        token_loss = self.token_loss(*tokens)
        recons_loss = F.mse_loss(recons, targets, reduction='sum') / len(targets)

        metrics = {
            f"{phase}/loss": token_loss,
            f"{phase}/token_xent": token_loss,
            f"{phase}/reconstruction_term": recons_loss,
        }

        return recons, slots, tokens, metrics

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ):
        _, _, _, metrics = self._step(batch, batch_idx, "train")

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
        _, slots, tokens, metrics = self._step(batch, batch_idx, phase)

        if (phase == "test") or (phase == "val" and batch_idx < self._ar_val_batches):
            target_tokens, target_images = tokens[1], batch[1]

            ar_recons, sampled_tokens = self.autoregressive_recons(slots)

            ar_recons_loss = nn.functional.mse_loss(
                ar_recons, target_images, reduction='sum'
            ) / len(target_images)
            ar_token_xent = self.token_loss(sampled_tokens, target_tokens)

            metrics[f"{phase}/ar_reconstruction_term"] = ar_recons_loss
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
        with torch.no_grad():
            _, token_output, *_ = self.backbone(inputs)
            token_idxs = self._get_token_idxs(token_output)
        tokens = self.token_dict(token_idxs).flatten(1, 2)
        return self.slot(tokens)

    def autoregressive_recons(self, slots):
        with torch.no_grad():
            sampled_discrete = self.sample_tokens(*slots).to(dtype=torch.float32)
            recons = self._decode_tokens(sampled_discrete)
            return recons, sampled_discrete

    def sample_tokens(self, slots, attn_weights=None):
        H, W = self.resolution
        slot_proj = self.slot_out_proj(slots)

        if attn_weights is not None and self.use_memory_mask:
            mask = attn_weights.detach()
        else:
            mask = None

        sampled_discrete = []
        token_inputs = self.bos_token.expand(len(slots), -1, -1)

        for pos in product(range(H), range(W)):
            u = self.transformer_decoder(token_inputs, slot_proj, mask)[:, -1:]
            new_token_idx = to_onehot(self.token_logits(u))

            sampled_discrete.append(new_token_idx)

            new_token = self.token_dict(new_token_idx, pos=pos)
            token_inputs = torch.cat([token_inputs, new_token], dim=1)

        return torch.cat(sampled_discrete, dim=1)


#---------------------------------------- Control Models -----------------------------------------

class SlotDecoderControl(BaseModel):
    def __init__(
        self,
        input_size: tuple[int, int, int],
        encoder_config: list,
        decoder_config: list,
        training: TrainingInit,
        n_slots: int = 4,
        slot_size: int = 64,
        n_iter: int = 3,
        slot_channels: int = 1,
        slot_hidden_size: int = 128,
        approx_implicit_grad: bool = False,
    ):
        super().__init__(training)
        self.save_hyperparameters()

        # Build encoder from config
        self.encoder = create_sequential(input_size, encoder_config)

        # Get encoder output size for slot attention input
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_size)
            encoder_output = self.encoder(dummy_input)
            encoder_output_size = encoder_output.shape[-1]

        # Build slot attention with explicit parameters
        self.slot_attention = SlotAttention(
            input_size=encoder_output_size,
            n_slots=n_slots,
            slot_size=slot_size,
            n_iter=n_iter,
            slot_channels=slot_channels,
            hidden_size=slot_hidden_size,
            approx_implicit_grad=approx_implicit_grad
        )

        # Build decoder from config (regular decoder, not SlotDecoder)
        decoder_input_size = n_slots * slot_size
        self.decoder = create_sequential(decoder_input_size, decoder_config)

        # Fixed MSE reconstruction loss
        self.recons_loss = nn.MSELoss(reduction='sum')

    def forward(
        self,
        inputs: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        h = self.encoder(inputs)
        slots, masks = self.slot_attention(h)
        recons = self.decoder(slots.flatten(1))
        return recons, (slots, masks)

    def _step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        phase: Literal["train", "val", "test"]
    ):
        inputs, targets = batch
        targets = 2 * targets - 1
        recons, (_, _) = self.forward(inputs)

        loss = self.recons_loss(recons, targets)

        is_train = phase == "train"
        self.log(
            f"{phase}/loss",
            loss,
            on_epoch=not is_train,
            on_step=is_train,
            prog_bar=is_train,
            sync_dist=not is_train,
            rank_zero_only=True
        )

        return loss


class SlotAttentionControl(BaseModel):
    def __init__(
        self,
        input_size: tuple[int, int, int],
        encoder_config: list,
        decoder_config: list,
        training: TrainingInit,
        n_slots: int = 4,
        slot_size: int = 64,
        beta: float = 1.0,
    ):
        super().__init__(training)
        self.save_hyperparameters()

        self.n_slots = n_slots
        self.slot_size = slot_size

        # Build encoder from config
        self.encoder = create_sequential(input_size, encoder_config)

        # Get encoder output size for latent layer input
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_size)
            encoder_output = self.encoder(dummy_input)
            encoder_output_size = encoder_output.shape[-1]

        # Build latent layer (DiagonalGaussian) - output size is n_slots * slot_size
        latent_size = n_slots * slot_size
        self.latent = DiagonalGaussian(encoder_output_size, latent_size)

        # Build decoder from config
        self.decoder = create_sequential(slot_size, decoder_config)

        # Fixed MSE reconstruction loss and KL regularization weight
        self.recons_loss = nn.MSELoss(reduction='sum')
        self.beta = beta

    def decode(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode slots into reconstructions and masks."""
        batch_size, n_slots = slots.size()[:2]

        # Batchify reconstruction - flatten slots for processing
        slots_flat = slots.flatten(end_dim=1)

        # Pass through decoder network to get RGBA output
        rgba = self.decoder(slots_flat)

        # Reshape back to (batch_size, n_slots, channels, height, width)
        rgba = rgba.unflatten(0, (batch_size, n_slots))

        # Split into reconstructions and mask logits (last channel is mask)
        slot_recons, slot_mask_logits = torch.tensor_split(rgba, indices=[-1], dim=2)

        # Apply softmax to mask logits across slots dimension
        slot_masks = torch.softmax(slot_mask_logits, dim=1)

        # Combine slot reconstructions weighted by masks
        recons = (slot_masks * slot_recons).sum(dim=1)

        return recons, slot_masks

    def forward(
            self,
            inputs: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        h = self.encoder(inputs)
        z, params = self.latent(h)
        slots = z.unflatten(1, (self.n_slots, self.slot_size))
        recons, slot_masks = self.decode(slots)
        return (recons, slot_masks), (slots, params)

    def _step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        phase: Literal["train", "val", "test"]
    ):
        inputs, targets = batch
        targert = 2 * targets - 1

        (recons, _), (_, params) = self.forward(inputs)

        recons_loss = self.recons_loss(recons, targets)

        # KL divergence loss for DiagonalGaussian
        mu, log_var = params
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1).mean()
        latent_loss = self.beta * kl_loss

        loss = recons_loss + latent_loss

        is_train = phase == "train"
        self.log_dict(
            {
                f"{phase}/loss": loss,
                f"{phase}/latent_term": latent_loss,
                f"{phase}/reconstruction_loss": recons_loss,
                f"{phase}/kl_loss": kl_loss,
            },
            on_epoch=not is_train,
            on_step=is_train,
            prog_bar=is_train,
            sync_dist=not is_train,
            rank_zero_only=True
        )

        return loss

    def predict(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self(inputs)

    def reconstruction(self, inputs):
        output = (self.predict(inputs)[0] + 1) / 2
        return output.clip(0, 1)


class FigureGroundDecoderControl(BaseModel):
    """
    Control model using FigureGroundSegmentation but with a standard non-slot decoder.
    The decoder takes the figure representation directly and reconstructs the image.
    """
    def __init__(
        self,
        input_size: tuple[int, int, int],
        encoder_config: list,
        decoder_config: list,
        training: TrainingInit,
        slot_size: int = 64,
        n_iter: int = 3,
        slot_channels: int = 1,
        slot_hidden_size: int = 128,
        approx_implicit_grad: bool = False,
    ):
        super().__init__(training)
        self.save_hyperparameters()

        # Build encoder from config
        self.encoder = create_sequential(input_size, encoder_config)

        # Get encoder output size for figure-ground segmentation input
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_size)
            encoder_output = self.encoder(dummy_input)
            encoder_output_size = encoder_output.shape[-1]

        # Build figure-ground segmentation
        self.fig_rep = FigureGroundSegmentation(
            input_size=encoder_output_size,
            latent_size=slot_size,
            n_iter=n_iter,
            n_channels=slot_channels,
            hidden_size=slot_hidden_size,
            approx_implicit_grad=approx_implicit_grad
        )

        # Build decoder from config (takes slot_size as input)
        self.decoder = create_sequential(slot_size, decoder_config)

        # Fixed MSE reconstruction loss
        self.recons_loss = nn.MSELoss(reduction='sum')

    def forward(
            self,
            inputs: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        h = self.encoder(inputs)
        fig_reps, masks = self.fig_rep(h)
        # fig_reps is (batch_size, 1, slot_size), squeeze to (batch_size, slot_size)
        recons = self.decoder(fig_reps.squeeze(1))
        return recons, (fig_reps, masks)

    def _step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        phase: Literal["train", "val", "test"]
    ):
        inputs, targets = batch
        targets = 2 * targets - 1

        recons, (_, _) = self.forward(inputs)

        loss = self.recons_loss(recons, targets) / len(targets)

        is_train = phase == "train"
        self.log(
            f"{phase}/loss",
            loss,
            on_epoch=not is_train,
            on_step=is_train,
            prog_bar=is_train,
            sync_dist=not is_train,
            rank_zero_only=True
        )

        return loss

    def predict(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self(inputs)

    def embed(self, inputs: torch.Tensor) -> torch.Tensor:
        h = self.encoder(inputs)
        fig_reps, _ = self.fig_rep(h)
        return fig_reps

    def reconstruction(self, inputs):
        output = (self.predict(inputs)[0] + 1) / 2
        return output.clip(0, 1)


class FigureGroundControl(BaseModel):
    """
    Control model using DiagonalGaussian (VAE) with figure-ground style decoding.
    Uses a single latent vector (like FigureGroundSegmentation) but sampled from a VAE.
    """
    def __init__(
        self,
        input_size: tuple[int, int, int],
        encoder_config: list,
        decoder_config: list,
        training: TrainingInit,
        latent_size: int = 64,
        beta: float = 1.0,
    ):
        super().__init__(training)
        self.save_hyperparameters()

        self.latent_size = latent_size

        # Build encoder from config
        self.encoder = create_sequential(input_size, encoder_config)

        # Get encoder output size for latent layer input
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_size)
            encoder_output = self.encoder(dummy_input)
            encoder_output_size = encoder_output.shape[-1]

        # Build latent layer (DiagonalGaussian) - single figure representation
        self.latent = DiagonalGaussian(encoder_output_size, latent_size)

        # Build decoder from config
        self.decoder = create_sequential(latent_size, decoder_config)

        # Fixed MSE reconstruction loss and KL regularization weight
        self.recons_loss = nn.MSELoss(reduction='sum')
        self.beta = beta

    def decode(self, fig_rep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode figure representation into reconstruction and mask."""
        # Pass through decoder network to get RGBA output
        rgba = self.decoder(fig_rep)

        # Split into reconstruction and mask logits (last channel is mask)
        fig_recons, fig_mask_logits = torch.tensor_split(rgba, indices=[-1], dim=1)

        # Apply sigmoid to mask logits (binary mask for figure vs background)
        fig_masks = torch.sigmoid(fig_mask_logits)

        # Combine figure reconstruction weighted by mask
        recons = fig_masks * fig_recons

        return recons, fig_masks

    def forward(
        self,
        inputs: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        h = self.encoder(inputs)
        z, params = self.latent(h)
        recons, fig_masks = self.decode(z)
        return (recons, fig_masks), (z, params)

    def _step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        phase: Literal["train", "val", "test"]
    ):
        inputs, targets = batch
        targets = 2 * targets - 1

        (recons, _), (_, params) = self.forward(inputs)

        recons_loss = self.recons_loss(recons, targets) / len(targets)

        # KL divergence loss for DiagonalGaussian
        mu, log_var = params
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1).mean()
        latent_loss = self.beta * kl_loss

        loss = recons_loss + latent_loss

        is_train = phase == "train"
        self.log_dict(
            {
                f"{phase}/loss": loss,
                f"{phase}/latent_term": latent_loss,
                f"{phase}/reconstruction_loss": recons_loss,
                f"{phase}/kl_loss": kl_loss,
            },
            on_epoch=not is_train,
            on_step=is_train,
            prog_bar=is_train,
            sync_dist=not is_train,
            rank_zero_only=True
        )

        return loss

    def predict(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self(inputs)

    def embed(self, inputs: torch.Tensor) -> torch.Tensor:
        h = self.encoder(inputs)
        z, _ = self.latent(h)
        return z.unsqueeze(1)  # Add slot dimension for consistency

    def reconstruction(self, inputs):
        output = (self.predict(inputs)[0][0] + 1) / 2
        return output.clip(0, 1)

