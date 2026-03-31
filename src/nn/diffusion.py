import math
import torch
import torch.nn as nn
from .init import linear_init
from .spatial import PositionEmbedding2D


def linear_schedule(T: int, beta_start: float, beta_end: float) -> dict[str, torch.Tensor]:
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    return {'betas': betas, 'alphas': alphas, 'alphas_bar': alphas_bar}


def cosine_schedule(T: int, s: float = 0.008) -> dict[str, torch.Tensor]:
    t = torch.arange(T + 1, dtype=torch.float32)
    f = torch.cos((t / T + s) / (1.0 + s) * math.pi / 2.0) ** 2
    alphas_bar_raw = torch.clamp(f / f[0], min=1e-4, max=0.9999)
    betas = torch.clamp(1.0 - alphas_bar_raw[1:] / alphas_bar_raw[:-1], max=0.999)
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    return {'betas': betas, 'alphas': alphas, 'alphas_bar': alphas_bar}


def local_2d_mask(H: int, W: int, radius: int = 1) -> torch.Tensor:
    """Bool mask of shape (H*W, H*W); True = blocked (cannot attend)."""
    rows = torch.arange(H).repeat_interleave(W)
    cols = torch.arange(W).repeat(H)
    dr = (rows[:, None] - rows[None, :]).abs()
    dc = (cols[:, None] - cols[None, :]).abs()
    return torch.maximum(dr, dc) > radius


def sinusoidal_timestep_embedding(t: torch.Tensor, d_model: int) -> torch.Tensor:
    """t: (batch,) int -> (batch, d_model)."""
    half = d_model // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class DiffusionDenoiser(nn.Module):
    def __init__(
        self,
        spatial_size: tuple[int, int],
        d_model: int,
        n_head: int,
        num_layers: int,
        ffwd_dim: int | None = 192,
        dropout: float = 0.0,
        neighborhood_radius: int = 1,
    ) -> None:
        super().__init__()
        H, W = spatial_size

        self.d_model = d_model
        self.spatial_size = spatial_size

        self.pos_emb = PositionEmbedding2D(d_model, H, W, embed='cardinal')

        decoder_layer = nn.TransformerDecoderLayer(
            d_model, n_head, ffwd_dim, dropout,
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers, norm=nn.LayerNorm(d_model),
        )
        self.out_proj = nn.Linear(d_model, d_model)
        self.register_buffer('attn_mask', local_2d_mask(H, W, neighborhood_radius))
        self._reset_parameters()

    @property
    def num_layers(self):
        return len(self.decoder.layers)

    def _reset_parameters(self) -> None:
        linear_init(self.out_proj, activation=None)
        gain = (3 * self.num_layers) ** (-0.5)
        for layer in self.decoder.layers:
            linear_init(layer.self_attn.out_proj, activation=None, gain=gain)  # type: ignore
            linear_init(layer.multihead_attn.out_proj, activation=None, gain=gain)  # type: ignore
            linear_init(layer.linear2, activation=None, gain=gain)

    def forward(
        self,
        tokens: torch.Tensor,   # (batch, H*W, d_model)
        memory: torch.Tensor,    # (batch, n_tokens, token_dim)
        t: torch.Tensor,        # (batch,) int
    ) -> torch.Tensor:          # (batch, H*W, d_model)
        H, W = self.spatial_size

        tgt = self.pos_emb(tokens.unflatten(1, (H, W))).flatten(1, 2)
        tgt = tgt + sinusoidal_timestep_embedding(t, self.d_model).unsqueeze(1)
        tgt = self.decoder(tgt, memory, tgt_mask=self.attn_mask, tgt_is_causal=False)

        return self.out_proj(tgt)
