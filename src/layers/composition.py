from abc import abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers.stochastic import cosine_decay


class CompositionOp(nn.Module):
    def __init__(self, latent_size: int, n_actions: int) -> None:
        super().__init__()
        assert latent_size > 0
        assert n_actions > 0
        assert n_actions <= latent_size

        self.latent_size = latent_size
        self.n_actions = n_actions

    @abstractmethod
    def __call__(
        self,
        inputs: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError


class SoftmaxComp(CompositionOp):
    def __init__(self, latent_size, n_actions) -> None:
        super().__init__(latent_size, n_actions)
        self.projection = nn.Sequential(
            nn.Linear(latent_size * 2 + n_actions, 4 * latent_size),
            nn.ReLU(),
            nn.Linear(4 * latent_size, latent_size)
        )

    def forward(self, z, action):
        all_inputs = torch.cat([z.flatten(1, 2), action], dim=1)
        logits = self.projection(all_inputs)
        return F.softmax(logits, dim=1)


class FixedInterpolationComp(CompositionOp):
    def __init__(self, latent_size, n_actions):
        super().__init__(latent_size, n_actions)
        self.latent_size = latent_size
        self.n_dummy = latent_size - n_actions

    def forward(self, z: torch.Tensor, actions):
        batch_size = len(actions)

        dummy_vars = actions.new_zeros((batch_size, self.n_dummy))
        actions = torch.cat([actions, dummy_vars], dim=1)

        z_ref, z_trans = z.chunk(2, 1)
        z_ref, z_trans = z_ref.squeeze(1), z_trans.squeeze(1)

        return z_ref * (1 - actions) + z_trans * actions


class LinearComp(CompositionOp):
    def __init__(self, latent_size, n_actions):
        super().__init__(latent_size, n_actions)
        self.latent_size = latent_size
        self.n_actions = n_actions
        self.ref_proj = nn.Linear(n_actions + latent_size, latent_size)
        self.trans_proj = nn.Linear(n_actions + latent_size, latent_size)

    def forward(self, z, actions):
        z_ref, z_trans = z.chunk(2, 1)

        z_ref = torch.cat([z_ref.squeeze(1), actions], dim=1).contiguous()
        z_trans = torch.cat([z_trans.squeeze(1), actions], dim=1).contiguous()

        return self.ref_proj(z_ref) + self.trans_proj(z_trans)


class InterpolationComp(CompositionOp):
    def __init__(self, latent_size, n_actions):
        super().__init__(latent_size, n_actions)
        self.linear = nn.Linear(2 * latent_size + n_actions, latent_size)

    def forward(self, z, actions):
        z_ref, z_trans = z.chunk(2, 1)
        z_ref, z_trans = z_ref.squeeze(1), z_trans.squeeze(1)

        w = self.linear(torch.cat([z.flatten(1, 2), actions], dim=1)).sigmoid()
        return z_ref * w + z_trans * (1.0 - w)


class MLPComp(CompositionOp):
    def __init__(self, latent_size, n_actions):
        super().__init__(latent_size, n_actions)
        input_size = 2 * latent_size + n_actions

        self.projection = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, latent_size)
        )

    def forward(self, z, actions):
        zpa = torch.cat([z.reshape(-1, self.latent_size), actions], dim=1)
        return self.projection(zpa.contiguous())


class SlotComposition(CompositionOp):
    def __init__(
        self,
        latent_size,
        n_actions,
        n_slots,
        tau,
        tau_start,
        tau_steps=None
    ):
        self.slot_select = nn.Sequential(
            nn.Linear(n_slots * latent_size + n_actions, 64),
            nn.ReLU(),
            nn.Linear(latent_size, n_slots)
        )

        self.projection = nn.Sequential(
            nn.Linear(2 * latent_size + n_actions, 64),
            nn.ReLU(),
            nn.Linear(64, latent_size)
        )

        if tau_steps is not None:
            self.tau_schedule = cosine_decay(tau_start, tau, tau_steps)
        else:
            self.tau_schedule = None
        self.step = -1

    def forward(self, slot_reps, actions):
        hard = not self.training

        if self.training and self.tau_schedule is not None:
            self.step += 1
            self.tau = self.tau_schedule(self.step)

        # split slots from orignal and transform images
        o_slots, t_slots = slot_reps.chunk(2, dim=1)

        # create input for action-depedant slot selction
        o_inputs = torch.cat((o_slots.flatten(1), actions), dim=-1)
        t_inputs = torch.cat((t_slots.flatten(1), actions), dim=-1)

        # compute probability that each slot contains particular object
        o_logits = F.log_softmax(self.slot_select(o_inputs), dim=-1)
        t_logits = F.log_softmax(self.slot_select(t_inputs), dim=-1)

        o_weights = F.gumbel_softmax(o_logits, self.tau, hard, dim=-1)
        t_weights = F.gumbel_softmax(t_logits, self.tau, hard, dim=-1)

        # sort weights so that selected slots are in position 0
        o_weights, o_idx = o_weights.sort(dim=1, descending=True)
        o_weights, t_idx = o_weights.sort(dim=1, dascending=True)

        # reorder input slots to match weights
        o_slots = o_slots.index_select(1, o_idx)
        t_slots = t_slots.index_select(1, t_idx)

        # soft selection of slots
        o_z = (o_slots * o_weights.unsqueeze(-1)).mean(1)
        t_z = (t_slots * t_weights.unsqueeze(-1)).mean(1)

        # transform slot
        # TODO: Need sparse recombination
        z = torch.cat((o_z, t_z, actions), dim=-1)
        transformed_slot = self.projection(z).unsqueeze(1)

        new_slots = torch.cat((transformed_slot, o_z[:, 1:]), dim=1)

        return new_slots
