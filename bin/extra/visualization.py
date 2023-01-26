from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from skimage import color

from .utils import IndexSelector, OutputSelector


class Visualzation(ABC):
    @abstractmethod
    def create_figure(self, model, loader, prefix):
        pass

    def set_owner(self, owner):
        self.owner = owner


class ReconstructionViz(Visualzation):
    def __init__(
        self,
        n_recons: int = 10,
        recons_transform: Literal["mse", "bce"] = "mse",
        recons_extractor: OutputSelector = IndexSelector(),
        include_inputs: bool = True,
    ):
        self.n_recons = n_recons
        self.recons_transform = recons_transform
        self.recons_extractor = recons_extractor
        self.include_inputs = include_inputs

    def __call__(self, model, loader,  prefix=""):
        return self.create_figure(model, loader,  prefix)

    @torch.no_grad()
    def create_figure(
        self,
        model: pl.LightningModule,
        loader: DataLoader,
        prefix: str
    ):
        inputs, recons = self.get_recons(model, loader)

        fig, axes = plt.subplots(
            2, self.n_recons, figsize=(2 * self.n_recons, 4))

        if self.include_inputs:
            images = np.stack([inputs.numpy(), recons.numpy()])
        else:
            images = recons.numpy()

        for j, img in enumerate(images):
            for i, img in enumerate(img):
                if np.prod(img.shape) == 3 * 64 * 64:
                    axes[j, i].imshow(img.reshape(3, 64, 64).transpose(1, 2, 0))
                else:
                    axes[j, i].imshow(img.reshape(1, 64, 64).transpose(1, 2, 0),
                                      cmap='Greys_r')

        for ax in axes.reshape(-1):
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        axes[0, 0].set_ylabel('input', fontsize=20)
        axes[1, 0].set_ylabel('recons', fontsize=20)

        fig_name = f"{prefix}_reconstructions"

        return fig_name, fig

    def get_recons(self, model, loader: DataLoader):

        inputs = sample_from_dataset(loader, self.n_recons)
        outputs = model(inputs.to(device=next(model.parameters()).device))

        recons = self.recons_extractor(outputs)

        if self.recons_transform == "mse":
            recons = recons.clamp(0, 1)
        else:
            recons = recons.sigmoid(0, 1)

        recons = recons.cpu()

        return inputs, recons


class SlotMaskViz(Visualzation):
    def __init__(
        self,
        n_samples: int = 10,
        mask_extractor: OutputSelector = IndexSelector(),
        decoder_mask_extractor: OutputSelector = IndexSelector(),
        include_inputs: bool = True,
    ):
        self.n_samples = n_samples
        self.mask_extractor = mask_extractor
        self.decoder_mask_extractor = decoder_mask_extractor
        self.include_inputs = include_inputs

    def __call__(self, model, loader, prefix):
        return self.create_figure(model, loader, prefix)

    @torch.no_grad()
    def create_figure(self, model, loader, prefix):
        inputs, atten_mask, decoder_mask = self.compute_masks(model, loader)

        atten_mask_fig = self._plot(inputs, atten_mask)
        atten_mask_fig_name = f"{prefix}_atten_mask"

        output = [(atten_mask_fig_name, atten_mask_fig)]

        if decoder_mask is not None:
            decoder_mask_fig = self._plot(inputs, decoder_mask)
            decoder_mask_fig_name = f"{prefix}_atten_mask"

            output.append((decoder_mask_fig_name, decoder_mask_fig))

        return output

    def compute_masks(self, model, loader):
        inputs = sample_from_dataset(loader, self.n_samples)
        H, W = inputs.shape[2:]

        outputs = model(inputs.to(device=model.device))

        inputs = inputs.permute(0, 2, 3, 1)

        atten_mask = self.mask_extractor(outputs)

        h = w = int(np.sqrt(atten_mask.shape[1]))

        # to (S, B, H, W)
        atten_mask = atten_mask.unflatten(1, (h, w)).permute(0, 3, 1, 2)

        # upsample
        atten_mask = atten_mask.unsqueeze_(-1).repeat_interleave(
            H // h, dim=2).repeat_interleave(W // w, dim=3)

        # get pixel level decoder mask
        if self.decoder_mask_extractor is not None:
            decoder_mask = self.decoder_mask_extractor(outputs)
            decoder_mask = decoder_mask.permute(0, 1, 3, 4, 2)
        else:
            decoder_mask = None

        return inputs, atten_mask, decoder_mask

    def _plot(self, inputs, masks):
        batch_size, n_slots = masks.shape[0], masks.shape[1]

        inputs = inputs.numpy()
        masks = masks.repeat(1, 1, 1, 1, 3).transpose(0, 1).numpy()

        masks = colorize(masks)

        if self.include_inputs:
            images = np.concatenate([inputs[np.newaxis], masks])
        else:
            images = masks

        labels = ['original'] + ['slot {}'.format(str(i+1)) for i in range(n_slots)]

        fig, axes = plt.subplots(
            n_slots + 1, batch_size,
            figsize=(2 * batch_size, 4 * (n_slots + 1))
        )

        for j, examp_imgs in enumerate(images):
            axes[j, 0].set_ylabel(labels[j], fontsize=20)

            for i, img in enumerate(examp_imgs):
                axes[j, i].imshow(img)

        for ax in axes.reshape(-1):
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        return fig


def sample_from_dataset(loader, n_samples):
    idx = np.random.choice(len(loader.dataset), size=n_samples, replace=False)
    inputs = torch.from_numpy(np.stack([loader.dataset[i][0] for i in idx]))
    return inputs


def colorize(masks):
    n_slots, batch_size = masks.shape[:2]
    hue_rotations = np.linspace(0, 1, n_slots + 1)

    def tint(image, hue, saturation=1):
        hsv = color.rgb2hsv(image)
        hsv[:, :, 1] = saturation
        hsv[:, :, 0] = hue
        return color.hsv2rgb(hsv)

    for i, h in zip(range(n_slots), hue_rotations):
        for j in range(batch_size):
            masks[i, j] = tint(masks[i, j], h)

    return masks
