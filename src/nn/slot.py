import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from .init import linear_init, gru_init


EPS = 1e-8


def join_heads(input: Tensor) -> Tensor:
    """
    Join attention heads of each slot. Assume heads are at dimension 1.
    """
    # input size: B, n_head, n_in, slot_size/input_size // n_head
    return input.transpose(1, 2).flatten(start_dim=2)


def split_heads(input: Tensor, n_heads: int) -> Tensor:
    """
    Split slots into n heads and set them to dimension 1.
    """
    # input size: B, N_in, (slot_size or input_size)
    split_size = input.shape[-1] // n_heads
    # output size B, H, N_in, I_s // H
    return input.unflatten(-1, (n_heads, split_size)).transpose(1, 2)


class SlotAttention(nn.Module):
    def __init__(
        self,
        input_size: int,
        n_slots: int = 4,
        slot_size: int = 64,
        n_iter: int = 3,
        slot_channels=1,
        hidden_size: int = 128,
        approx_implicit_grad: bool = True
    ) -> None:
        super().__init__()

        assert n_slots > 1, "Must have at least two slots"
        assert n_iter > 0, "Need at least one slot update iteration"
        assert (slot_size % slot_channels) == 0

        self.n_slots = n_slots
        self.slot_size = slot_size
        self.n_iter = n_iter
        self.nhead = slot_channels
        self.approx_implicit_grad = approx_implicit_grad

        self.slot_mu = Parameter(torch.empty(1, 1, slot_size))
        self.slot_logvar = Parameter(torch.empty(1, 1, slot_size))

        self.k_proj = nn.Linear(input_size, slot_size, bias=False)
        self.v_proj = nn.Linear(input_size, slot_size, bias=False)
        self.q_proj = nn.Linear(slot_size, slot_size, bias=False)

        self.norm_input = nn.LayerNorm(input_size)
        self.norm_slot = nn.LayerNorm(slot_size)
        self.norm_res = nn.LayerNorm(slot_size)

        self.gru = nn.GRUCell(slot_size, slot_size)
        self.mlp = nn.Sequential(
            nn.Linear(slot_size, hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_size, slot_size)
        )

        self.reset_parameters()

    @property
    def size(self):
        return self.n_slots * self.slot_size

    @property
    def hidden_size(self):
        return self.mlp[0].out_features

    @property
    def shape(self):
        return self.n_slots, self.slot_size

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.slot_mu)
        nn.init.xavier_uniform_(self.slot_logvar)

        linear_init(self.k_proj, activation='relu')
        linear_init(self.v_proj, activation='relu')
        linear_init(self.q_proj, activation='relu')

        for m in self.mlp.children():
            if isinstance(m, nn.Linear):
                linear_init(m, activation='relu')

        gru_init(self.gru)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        init_slot = slots = self.init_slots(inputs)  # batch_size, n_slots, slot_size
        inputs = self.norm_input(inputs)  # batch_size, n_inputs, input_size

        # shape (batch_size, n_heads, n_inputs, slot_size // nheads)
        k = split_heads(self.k_proj(inputs), self.nhead) / ((self.slot_size / self.nhead) ** 0.5)
        v = split_heads(self.v_proj(inputs), self.nhead)

        # NOTE: We approximate the implicit gradient using a first-order Neumann method. Notice
        # that this is an approximation because we are not performing a full backward pass nor
        # are we using a root-finding algorithm to find the JVP as in DEQs. See Chang et al.
        with torch.set_grad_enabled(not self.approx_implicit_grad):
            for _ in range(self.n_iter):
                slots, atten_masks = self.step(slots, k, v)

        # Re-engage the gradient tape if we disabled it. Note that we must make the gradient
        # pass through to the initial representation state, which is a parameter of the model
        if self.approx_implicit_grad:
            slots = (slots - init_slot).detach() + init_slot  # pass-through gradient to init_slot
            slots, atten_masks = self.step(slots, k, v)

        return slots, atten_masks.sum(dim=1)

    def init_slots(self, inputs):
        std = self.slot_logvar.mul(0.5).exp()
        std = std.expand(len(inputs), self.n_slots, -1)
        eps = torch.randn_like(std)
        return self.slot_mu.addcmul(std, eps)

    def step(self, slots, k, v):
        q = self.q_proj(self.norm_slot(slots))
        # atten_maps: (batch_sizs, n_slots, slot_size)
        # atten_weights: (batch_size, n_heads, n_slots, slot_size // n_heads)
        atten_maps, atten_weights = self.compute_attention_maps(k, q, v)
        slots = self.update_slots(atten_maps, slots)
        return slots, atten_weights

    def compute_attention_maps(self, k, q, v):
        q = split_heads(q, self.nhead)

        # k: b, h, n_in, e; q: b, h, s, e
        weights = k @ q.transpose(2, 3)  # b, h, n_inputs, n_slots
        # softmax over slots and heads
        weights = F.softmax(join_heads(weights), dim=-1)
        # split back to b, h, n_in, s
        weights = split_heads(weights, self.nhead) + EPS
        weights = weights / weights.sum(dim=-2, keepdim=True)

        atten_maps = join_heads(weights.transpose(2, 3) @ v)
        return atten_maps, weights

    def update_slots(self, atten_maps, slots):
        B = len(slots)
        # batchify update
        atten_maps = atten_maps.flatten(end_dim=1)
        slots = slots.flatten(end_dim=1)
        slots = self.gru(atten_maps, slots)
        slots = slots + self.mlp(self.norm_res(slots))
        return slots.unflatten(0, (B, self.n_slots))

    def __repr__(self):
        return 'SlotAttention(n_slots={}, slot_size={}, n_iter={})'.format(
            self.n_slots, self.slot_size, self.n_iter)


class FigureGroundSegmentation(nn.Module):
    """
    Figure-ground segmentation based on the Slot Attention model.

    Instead of slots competing amongst each other for input assignments,
    a single latent vector is tasked with representing the patches that are
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
        return 1, self.latent_size

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

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        init_figure_rep = figure_rep = self.init_fig_rep(inputs)  # batch_size, latent_size
        inputs = self.norm_input(inputs)  # batch_size, n_inputs, input_size

        # shape (batch_size, n_heads, n_inputs, latent_size // nheads)
        k = split_heads(self.k_proj(inputs), self.nhead) / ((self.latent_size / self.nhead) ** 0.5)
        v = split_heads(self.v_proj(inputs), self.nhead)

        # NOTE: We approximate the implicit gradient using a first-order Neumann method. Notice
        # that this is an approximation because we are not performing a full backward pass nor
        # are we using a root-finding algorithm to find the JVP as in DEQs. See Chang et al.
        with torch.set_grad_enabled(not self.approx_implicit_grad):
            for _ in range(self.n_iter):
                figure_rep, atten_masks = self.step(figure_rep, k, v)

        # Re-engage the gradient tape if we disabled it. Note that we must make the gradient
        # pass through to the initial representation state, which is a parameter of the model
        if self.approx_implicit_grad:
            figure_rep = (figure_rep - init_figure_rep).detach() + init_figure_rep
            figure_rep, atten_masks = self.step(figure_rep, k, v)

        return figure_rep.unsqueeze(1), atten_masks.sum(dim=1)  #type: ignore

    def init_fig_rep(self, inputs):
        std = self.init_logvar.mul(0.5).exp().expand(len(inputs), -1)
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
        weights = torch.sigmoid(weights)
        weights = weights / (weights.sum(dim=-2, keepdim=True) + EPS)

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
        return 'FigureGroundSegmentation(latent_size={}, n_iter={})'.format(
            self.latent_size, self.n_iter)


class FigureGroundSegmentationV2(nn.Module):
    """
    Figure-ground segmentation based on the Slot Attention model.

    Instead of slots competing amongst each other for input assignments,
    a single latent vector is tasked with representing the patches that are
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
        self.virtual_slot = Parameter(torch.empty(1, latent_size))

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
        return 1, self.latent_size

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

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        init_figure_rep = figure_rep = self.init_fig_rep(inputs)  # batch_size, latent_size
        inputs = self.norm_input(inputs)  # batch_size, n_inputs, input_size

        # shape (batch_size, n_heads, n_inputs, latent_size // nheads)
        k = split_heads(self.k_proj(inputs), self.nhead) / ((self.latent_size / self.nhead) ** 0.5)
        v = split_heads(self.v_proj(inputs), self.nhead)

        # NOTE: We approximate the implicit gradient using a first-order Neumann method. Notice
        # that this is an approximation because we are not performing a full backward pass nor
        # are we using a root-finding algorithm to find the JVP as in DEQs. See Chang et al.
        with torch.set_grad_enabled(not self.approx_implicit_grad):
            for _ in range(self.n_iter):
                figure_rep, atten_masks = self.step(figure_rep, k, v)

        # Re-engage the gradient tape if we disabled it. Note that we must make the gradient
        # pass through to the initial representation state, which is a parameter of the model
        if self.approx_implicit_grad:
            figure_rep = (figure_rep - init_figure_rep).detach() + init_figure_rep
            figure_rep, atten_masks = self.step(figure_rep, k, v)

        return figure_rep.unsqueeze(1), atten_masks.sum(dim=1)  #type: ignore

    def init_fig_rep(self, inputs):
        std = self.init_logvar.mul(0.5).exp().expand(len(inputs), -1)
        eps = torch.randn_like(std)
        return self.init_mu.addcmul(std, eps)

    def step(self, fig_rep, k, v):
        q = self.q_proj(self.norm_slot(fig_rep)).unsqueeze(1)
        q = torch.cat([self.virtual_slot.expand(len(q), 1, -1), q], dim=1)  # add the virtual slot
        # atten_maps: (batch_sizs, n_slots, slot_size)
        # atten_weights: (batch_size, n_heads, n_slots, slot_size // n_heads)
        atten_maps, atten_weights = self.compute_attention_maps(k, q, v)

        # Note: we let the virtual_slot compete withe the active slot, but don't update it.
        fig_atten_map, fig_atten_weights = atten_maps[:, 1], atten_weights[:, :, 1]
        fig_rep = self.update_latent(fig_atten_map, fig_rep)
        return fig_rep, fig_atten_weights

    def compute_attention_maps(self, k, q, v):
        q = split_heads(q, self.nhead)
        # k: b, h, n_in, e; q: b, h, s, e
        weights = k @ q.transpose(2, 3)  # b, h, n_inputs, n_slots
        # softmax over slots and heads
        weights = F.softmax(join_heads(weights), dim=-1)
        # split back to b, h, n_in, s
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
        return 'FigureGroundSegmentation(latent_size={}, n_iter={})'.format(
            self.latent_size, self.n_iter)
