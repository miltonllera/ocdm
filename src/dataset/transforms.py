import torch
import torch.nn as nn


class RescaleTargets(nn.Module):
    def __init__(self, mean, min, max) -> None:
        super().__init__()
        assert len(mean) == len(min) == len(max)
        self.mean = torch.asarray(mean)
        self.min = torch.asarray(min)
        self.max = torch.asarray(max)

    def __call__(self, inputs):
        return (torch.asarray(inputs) - self.mean) / (self.max - self.min)
