from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Union

import numpy as np
import torch
from numpy.random.mtrand import choice
from torch import Tensor, cat


def collate(inputs):
    return torch.from_numpy(np.stack(inputs))


def random_sample(loader, n_samples):
    idx = np.random.choice(len(loader.dataset), size=n_samples, replace=False)
    data_points = [loader.dataset[i] for i in idx]
    data_points = tuple(map(collate, zip(*data_points)))
    return data_points


def grid_sample(loader, n_samples):
    step = len(loader.dataset) // n_samples
    idx = range(0, len(loader.dataset), step)
    data_points = [loader.dataset[i] for i in idx]
    data_points = tuple(map(collate, zip(*data_points)))
    return data_points


def subsample(*arrays, n_samples=10000):
    idx = choice(len(arrays[0]), size=n_samples)
    subsampled = [a[idx] for a in arrays]
    return subsampled


@torch.no_grad()
def compute_embeddings(model, loader):
    embeddings = []
    targets = []

    for x, y in loader:
        x = x.to(device=model.device)
        e = model.embed(x)

        embeddings.append(e)
        targets.append(y)

    return torch.cat(embeddings), torch.cat(targets)


@torch.no_grad()
def compute_outputs(model, loader):
    reconstructions = []
    targets = []

    for x, y in loader:
        x = x.to(device=model.device)
        e = model(x)

        reconstructions.append(e)
        targets.append(y)

    return torch.cat(reconstructions), torch.cat(targets)


class OutputSelector(ABC):
    def __call__(self, model_outputs: Any) -> Any:
        return self.select(model_outputs)

    @abstractmethod
    def select(self, model_outputs: Any) -> Any:
        pass


class IndexSelector(OutputSelector):
    def __init__(self, idx: Union[int, List[int]] = 0) -> None:
        if isinstance(idx, int):
            idx = [idx]
        self.idx = idx

    def select(self, model_outputs):
        out = model_outputs
        for i in self.idx:
            out = out[i]
        return out


def create_output_transform(pred_transform):
    def output_transform(output):
        pred, target = output
        pred = pred_transform(pred)
        return pred, target
    return output_transform


SLOT_AE_OUTPUT = Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]

def collate_slotae_output(slot_ae_outputs: List[SLOT_AE_OUTPUT]) -> SLOT_AE_OUTPUT:
    recons, seg_mask, slots, attn_mask = [], [], [], []

    for (r, sm), (s, am) in slot_ae_outputs:
        recons.append(r)
        seg_mask.append(sm)
        slots.append(s)
        attn_mask.append(am)

    return (cat(recons), cat(seg_mask)), (cat(slots), cat(attn_mask))
