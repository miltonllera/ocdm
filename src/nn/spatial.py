import torch
import torch.nn as nn
from .init import linear_init


def cardinal_pos_embedding(resolution, min_val=0, max_val=1.0):
    grid = origin_pos_embedding(resolution, min_val, max_val)
    return torch.cat([grid, 1.0 - grid], dim=0)


def origin_pos_embedding(resolution, min_val=-1.0, max_val=1.0):
    ranges = [torch.linspace(min_val, max_val, steps=r) for r in resolution]
    grid = torch.meshgrid(*ranges, indexing='xy')
    grid = torch.stack(grid, dim=0)
    return grid.to(dtype=torch.float32)


def sine_pos_embedding(resolution, n_channels, temperature=10000):
    H, W = resolution
    # Standard 2D sinusoidal position embedding (e.g. DETR)
    num_pos_feats = n_channels // 2

    y_embed = torch.arange(1, H + 1, dtype=torch.float32)
    x_embed = torch.arange(1, W + 1, dtype=torch.float32)

    dim_t = torch.arange(num_pos_feats // 2, dtype=torch.float32)
    dim_t = temperature ** (2 * dim_t / num_pos_feats)

    pos_y = y_embed[:, None] / dim_t
    pos_x = x_embed[:, None] / dim_t

    pos_y = torch.stack([pos_y.sin(), pos_y.cos()], dim=2).flatten(1)
    pos_x = torch.stack([pos_x.sin(), pos_x.cos()], dim=2).flatten(1)

    pos_y = pos_y.unsqueeze(1).repeat(1, W, 1)
    pos_x = pos_x.unsqueeze(0).repeat(H, 1, 1)

    grid = torch.cat([pos_y, pos_x], dim=-1)

    if grid.shape[-1] < n_channels:
        padding = torch.zeros(H, W, n_channels - grid.shape[-1])
        grid = torch.cat([grid, padding], dim=-1)

    return grid


class PositionConcat(nn.Module):
    def __init__(self, height, width=None, dim=-3, embed='origin'):
        super().__init__()

        if width is None:
            width = height

        self.height = height
        self.width = width

        if embed == 'cardinal':
            grid = cardinal_pos_embedding((height, width))
        elif embed == 'origin':
            grid = origin_pos_embedding((height, width))
        else:
            raise ValueError('Unrecognized embedding type {}'.format(embed))

        self.grid = grid
        self.dim = dim

    def forward(self, inputs):
        sizes = list(inputs.shape[:-3]) + [-1, -1, -1]
        grid = self.grid.expand(sizes).to(device=inputs.device)
        return torch.cat([inputs, grid], dim=self.dim).contiguous()

    def __repr__(self):
        return 'PositionConcat(height={},width={})'.format(self.height, self.width)


class PositionEmbedding1D(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(max_len, d_model), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.pe)

    def forward(self, input, start_pos=0):
        T = input.shape[1]
        return input + self.pe[start_pos:start_pos + T]


class PositionEmbedding2D(nn.Module):
    def __init__(self, n_channels, height, width=None, embed='cardinal'):
        super().__init__()

        if width is None:
            width = height

        self.height = height
        self.width = width

        if embed == 'cardinal':
            grid = cardinal_pos_embedding((height, width))
            linear = nn.Linear(4, n_channels)
            linear_init(linear, activation=None)
            self.grid = grid.transpose(2, 0)
            self.projection = linear
        elif embed == 'origin':
            grid = origin_pos_embedding((height, width))
            linear = nn.Linear(2, n_channels)
            linear_init(linear, activation=None)
            self.grid = grid.transpose(2, 0)
            self.projection = linear
        elif embed == 'sine':
            grid = sine_pos_embedding((height, width), n_channels)
            self.grid = grid
            self.projection = nn.Identity()
        else:
            raise ValueError('Unrecognized embedding type {}'.format(embed))

    def get_projection(self, device):
        return self.projection(self.grid.to(device=device))

    def forward(self, inputs: torch.Tensor, pos=None) -> torch.Tensor:
        proj = self.projection(self.grid.to(device=inputs.device))
        if pos is not None:
            proj = proj[*pos]
        return inputs + proj

    def reset_parameters(self):
        if isinstance(self.projection, nn.Linear):
            linear_init(self.projection, activation=None)

    def __repr__(self):
        return 'PositionEmbedding2D(height={}, width={})'.format(
                self.height, self.width)


class SpatialBroadcast(nn.Module):
    def __init__(self, height, width=None, input_last=False) -> None:
        super().__init__()

        if width is  None:
            width = height

        self.width = width
        self.height = height
        self.input_last = input_last

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        bdc = torch.tile(inputs[(..., None, None)], (self.height, self.width))
        if self.input_last:
            bdc = torch.movedim(bdc, -3, -1)
        return bdc

    def __repr__(self):
        return 'SpatialBroadcast(height={},width={}, input_last={})'.format(
                self.height, self.width, self.input_last)


# class WeightedSBC(SpatialBroadcast):
#     def forward(self, inputs: Tuple[Tensor, Tensor]) -> Tensor:
#         z, mask = inputs

#         # tiled, shape (bs * n_slots, slot_size, height, width)
#         tiled = torch.tile(z[(..., None, None)], (self.height, self.width))
#         mask = mask.unsqueeze(1).unflatten(-1, (self.height, self.width))

#         return tiled * mask
