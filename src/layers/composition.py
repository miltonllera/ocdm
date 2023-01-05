import torch
import torch.nn as nn
import torch.nn.functional as F


class CompositionOp(nn.Module):
    def __init__(self, latent_size: int, n_actions: int) -> None:
        super().__init__()
        assert latent_size > 0
        assert n_actions > 0
        assert n_actions <= latent_size

        self.latent_size = latent_size
        self.n_actions = n_actions


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
        self.projection = nn.Sequential(nn.Linear(input_size, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, latent_size))

    def forward(self, z, actions):
        zpa = torch.cat([z.reshape(-1, self.latent_size), actions], dim=1)
        return self.projection(zpa.contiguous())
