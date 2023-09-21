from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
from .initialization import linear_init, gru_init
from .slot import join_heads, split_heads, EPS


class FigureGroundSegmentation(nn.Module):
    """
    Figure-ground segmentation based on the Slot Attention model.

    Instead of slots competing amongst each other for input assignments, a
    single latent representation tries to represent the patches that are
    part of a target object (the figure) while ignoring the rest (background).

    The resulting model is much simpler than the original SA as it removes the
    need to track the usage of the slots.
    """
    def __init__(
        self,
        input_size: int,
        latent_size: int = 64,
        n_iter: int = 3,
        n_channels = 1,
        hidden_size = 128,
        approx_implicit_grad: bool = True,
    ) -> None:
        super().__init__()

        self.latent_size = latent_size
        self.n_iter = n_iter
        self.nhead = n_channels
        self.approx_implicit_grad = approx_implicit_grad

        self.init_mu = Parameter(torch.empty(1, latent_size))
        self.init_logvar = Parameter(torch.empty(1, latent_size))

        self.k_proj = nn.Linear(input_size, latent_size, bias=False)
        self.v_proj = nn.Linear(input_size, latent_size, bias=False)
        self.q_proj = nn.Linear(latent_size, latent_size, bias=False)

        self.norm_input = nn.LayerNorm(input_size)
        self.norm_slot = nn.LayerNorm(latent_size)
        self.norm_res = nn.LayerNorm(latent_size)

        self.gru = nn.GRUCell(latent_size, latent_size)
        self.mlp = nn.Sequential(
            nn.Linear(latent_size, hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size)
        )

        self.reset_parameters()

    @property
    def size(self):
        return self.latent_size

    @property
    def hidden_size(self):
        return self.mlp[0].out_features

    @property
    def shape(self):
        return self.n_slots, self.slot_size

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.init_mu)
        nn.init.xavier_uniform_(self.init_logvar)

        linear_init(self.k_proj, activation='relu')
        linear_init(self.v_proj, activation='relu')
        linear_init(self.q_proj, activation='relu')

        for m in self.mlp.children():
            if isinstance(m, nn.Linear):
                linear_init(m, activation='relu')

        gru_init(self.gru)

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        figure_rep = self.init_fig_rep(inputs)  # batch_size, latent_size
        inputs = self.norm_input(inputs)  # batch_size, n_inputs, input_size

        # shape (batch_size, n_heads, n_inputs, latent_size // nheads)
        k = split_heads(self.k_proj(inputs), self.nhead)
        v = split_heads(self.v_proj(inputs), self.nhead)

        k = k * (self.nhead ** 0.5)

        for _ in range(self.n_iter):
            figure_rep, atten_masks = self.step(figure_rep, k, v)

        # First-order Neumann approximation to implicit gradient (Chang et al)
        if self.approx_implicit_grad:
            figure_rep, atten_masks = self.step(figure_rep.detach(), k, v)

        return figure_rep, atten_masks.sum(dim=1)  #type: ignore [reportGeneralTypeIssues]

    def init_fig_rep(self, inputs):
        std = self.init_logvar.mul(0.5).exp()
        std = std.expand(len(inputs), -1)
        eps = torch.randn_like(std)
        return self.init_mu.addcmul(std, eps)

    def step(self, fig_rep, k, v):
        q = self.q_proj(self.norm_slot(fig_rep))

        # atten_maps: (batch_sizs, n_slots, slot_size)
        # atten_weights: (batch_size, n_heads, n_slots, slot_size // n_heads)
        atten_maps, atten_weights = self.compute_attention_maps(k, q, v)

        fig_rep = self.update_latent(atten_maps, fig_rep)

        return fig_rep, atten_weights

    def compute_attention_maps(self, k, q, v):
        q = split_heads(q.unsqueeze(1), self.nhead)

        weights = k @ q.transpose(2, 3)
        weights = torch.sigmoid(join_heads(weights))

        weights = split_heads(weights, self.nhead) + EPS
        weights = weights / weights.sum(dim=-2, keepdim=True)

        atten_maps = join_heads(weights.transpose(2, 3) @ v)

        return atten_maps, weights

    def update_latent(self, atten_maps, fig_rep):
        # batchify update
        atten_maps = atten_maps.squeeze(1)
        fig_rep = fig_rep.squeeze(1)

        fig_rep = self.gru(atten_maps, fig_rep)
        fig_rep = fig_rep + self.mlp(self.norm_res(fig_rep))

        return fig_rep

    def __repr__(self):
        return 'SlotAttention(n_slots={}, slot_size={}, n_iter={})'.format(
            self.n_slots, self.slot_size, self.n_iter)


class FigDecoder(nn.Sequential):
    def decode_fig(self, inputs):
        fig_reps, _ = inputs

        rgba = super().forward(fig_reps)

        fig_recons, fig_mask = torch.tensor_split(rgba, indices=[-1], dim=1)
        fig_mask = torch.sigmoid(fig_mask)  # masks are logits

        return fig_recons, fig_mask

    def forward(self, inputs: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        slot_recons, fig_rep = self.decode_fig(inputs)
        return (fig_rep * slot_recons), fig_rep
