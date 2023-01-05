from typing import Optional
import torch
import torch.nn as nn
from .initialization import linear_init


class TransformerDecoder(nn.TransformerDecoder):
    def __init__(
        self,
        max_seqlen: int,
        d_model: int=512,
        n_head: int=4,
        num_layers: int=4,
        ffwd_dim: Optional[int] = None,
        dropout: float=0.0,
        mask: Optional[torch.Tensor] = None,
    ):
        if ffwd_dim is None:
            ffwd_dim = 4 * d_model

        if mask is None:
            mask = autoregressive_mask(max_seqlen)

        layer_norm = nn.LayerNorm(d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model,
            n_head,
            ffwd_dim,
            dropout,
            batch_first=True,
            norm_first=True
        )

        super().__init__(decoder_layer, num_layers, layer_norm)

        self.d_model = d_model
        self.nhead = n_head
        self.mask = mask

        init_projections(self, num_layers)

    def forward(self, tgt, memory, memory_mask=None):
        L = tgt.shape[1]  # required for autoregressive generation

        mask = self.mask[:L, :L].to(device=tgt.device)
        if memory_mask is not None:
            # memory_mask must match batch_size * nhead for MultiheadAttention
            memory_mask = memory_mask[:,:L].unsqueeze(1).expand(
                    -1, self.nhead, -1, -1
                ).flatten(end_dim=1)

        return super().forward(tgt, memory, mask, memory_mask)


def autoregressive_mask(size):
    return torch.ones((size, size)).triu(diagonal=1).to(dtype=torch.bool)


def init_projections(transformer, num_layers):
    gain = (3 * num_layers) ** (-0.5)
    for mod in transformer.children():
        if isinstance(mod, nn.MultiheadAttention):
            linear_init(mod.out_proj, activation=None, gain=gain)
        if isinstance(mod, nn.TransformerDecoderLayer):
            linear_init(mod.linear2, activation=None, gain=gain)
