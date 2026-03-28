import torch.nn as nn


class PatchDiscriminator(nn.Sequential):
    def __init__(
        self,
        n_layers: int,
        first_layer_n_channles: int,
    ) -> None:
        layers, in_channels, out_channels = [], 3, first_layer_n_channles
        for _ in range(n_layers):
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 4, 2, 1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2)
            ])
            in_channels = out_channels
            out_channels *= 2

        layers.append(nn.Conv2d(in_channels, 1, 1, 1, 0))

        super().__init__(*layers)
