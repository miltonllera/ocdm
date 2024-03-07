import os.path  as osp
from abc import ABC, abstractmethod
from typing import Any, Callable, Literal, Optional

import numpy as np
# from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from skimage import color

from src.analysis.hinton import IndexLocator, hinton
from .utils import (
    IndexSelector,
    OutputSelector,
    random_sample,
    subsample,
    compute_embeddings
)


class Visualzation(ABC):
    def __call__(self, model, datamodule) -> Any:
        return self.create_figure(model, datamodule)

    @abstractmethod
    def create_figure(self, model, datamodule):
        pass

    def set_owner(self, owner):
        self.owner = owner


class ReconstructionViz(Visualzation):
    def __init__(
        self,
        n_recons: int = 3,
        recons_transform: Literal["mse", "bce"] = "mse",
        data_split: Literal["train", "test", "both"] = "both",
        recons_extractor: OutputSelector = IndexSelector(),
    ):
        self.n_recons = n_recons
        self.recons_transform = recons_transform
        self.data_split = data_split
        self.recons_extractor = recons_extractor

    @torch.no_grad()
    def create_figure(
        self,
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule,
    ):
        if self.data_split == "train":
            inputs, recons = self.get_recons(
                model,
                datamodule.val_dataloader(),
            )
        elif self.data_split == "test":
            inputs, recons = self.get_recons(
                model,
                datamodule.test_dataloader(),
            )
        else:
            train_inputs, train_recons = self.get_recons(
                model,
                datamodule.val_dataloader(),
            )
            test_inputs, test_recons = self.get_recons(
                model,
                datamodule.test_dataloader(),
            )

            inputs = torch.cat([train_inputs, test_inputs])
            recons = torch.cat([train_recons, test_recons])

        return self.plot(inputs, recons)

    def get_recons(self, model, loader: DataLoader):
        data_points = random_sample(loader, self.n_recons)

        inputs = data_points[0].to(device=next(model.parameters()).device)
        outputs = model(inputs)

        recons = self.recons_extractor(outputs)

        if self.recons_transform == "mse":
            recons = recons.clamp(0, 1)
        else:
            recons = recons.sigmoid_(0, 1)

        recons = recons.cpu()

        return inputs, recons

    def plot(self, inputs, recons):
        images = np.stack([inputs.numpy(), recons.numpy()])

        n_cols = ((self.data_split == "both") + 1) * self.n_recons
        fig, axes = plt.subplots(2, n_cols, figsize=(2 * self.n_recons, 2))

        images = images.transpose(0, 1, 3, 4, 2)
        if images.shape[-1] == 1:
            images = np.repeat(images, 3, -1)

        for j, sample in enumerate(images):
            for i, img in enumerate(sample):
                axes[j, i].imshow(img)

        for ax in axes.reshape(-1):
            strip_plot(ax)

        axes[0, 0].set_ylabel('input', fontsize=20)
        axes[1, 0].set_ylabel('recons', fontsize=20)

        if self.data_split == "both":
            axes[0, axes.shape[1] // 4].set_title("train")
            axes[0, 2 * axes.shape[1] // 4].set_title("test")
        else:
            axes[0, axes.shape[1] // 2].set_title(self.data_split)

        plt.tight_layout()

        return f"{self.data_split}-{self.n_recons}_recons", fig


class SlotReconstruction(Visualzation):
    def __init__(
        self,
        n_recons: int = 3,
        recons_transform: Literal["mse", "bce"] = "mse",
        data_split: Literal["train", "test", "both"] = "both",
        recons_extractor: OutputSelector = IndexSelector(),
        slot_mask_extractor: Optional[OutputSelector] = IndexSelector(),
        decoder_mask_extractor: Optional[OutputSelector] = IndexSelector(),
    ):
        self.n_recons = n_recons
        self.recons_transform = recons_transform
        self.data_split = data_split
        self.recons_extractor = recons_extractor
        self.slot_mask_extractor = slot_mask_extractor
        self.decoder_mask_extractor = decoder_mask_extractor
        self.name = "slot_ae_examples"

        # gridspec axes params
        self.H, self.W = 2, 2

    @torch.no_grad()
    def create_figure(self, model, datamodule):
        if self.data_split == "train":
            outputs = self.compute_outputs(
                model,
                datamodule.val_dataloader(),
            )
        elif self.data_split == "test":
            outputs = self.compute_outputs(
                model,
                datamodule.test_dataloader(),
            )
        else:
            train_outputs = self.compute_outputs(
                model,
                datamodule.val_dataloader(),
            )
            test_outputs = self.compute_outputs(
                model,
                datamodule.test_dataloader(),
            )

            outputs = []
            for train, test in zip(train_outputs, test_outputs):
                if train is not None and test is not None:
                    all_out = torch.cat([train, test])
                else:
                    all_out = None

                outputs.append(all_out)

        return self.plot(*outputs)

    def compute_outputs(self, model, loader):
        data_points = random_sample(loader, self.n_recons)

        inputs = data_points[0].to(device=model.device)
        H, W = inputs.shape[2:]

        outputs = model(inputs)
        inputs = inputs.permute(0, 2, 3, 1)
        recons = self.recons_extractor(outputs).permute(0, 2, 3, 1)

        if self.recons_transform == "mse":
            recons = recons.clamp(0, 1)
        else:
            recons = recons.sigmoid_()

        if self.slot_mask_extractor is not None:
            atten_mask = self.slot_mask_extractor(outputs)
            h = w = int(np.sqrt(atten_mask.shape[1]))

            # norm color
            atten_mask = atten_mask / atten_mask.sum(dim=-1, keepdim=True)

            # to (S, B, H, W)
            atten_mask = atten_mask.unflatten(1, (h, w)).permute(0, 3, 1, 2)

            # upsample
            atten_mask = atten_mask.unsqueeze_(-1).repeat_interleave(
                H // h, dim=2).repeat_interleave(W // w, dim=3)
        else:
            atten_mask = None

        # get pixel level decoder mask
        if self.decoder_mask_extractor is not None:
            decoder_mask = self.decoder_mask_extractor(outputs)

            if len(decoder_mask.shape) < 5:  # account for figure-ground segmentation
                decoder_mask = decoder_mask.unsqueeze_(1)

            decoder_mask = decoder_mask.permute(0, 1, 3, 4, 2)
        else:
            decoder_mask = None

        return inputs, atten_mask, decoder_mask, recons

    def plot(self, inputs, atten_masks, decoder_mask, recons):
        # Define grid specs parameters
        has_attn_mask = atten_masks is not None
        has_dec_mask = decoder_mask is not None

        n_slots = 0
        if has_attn_mask:
            n_slots = atten_masks.shape[1]
        elif has_dec_mask:
            n_slots = decoder_mask.shape[1]

        # create figure
        fig, gs = self.init_figure(n_slots, has_attn_mask, has_dec_mask)

        # plot inputs and reconstructions
        self.plot_reconstructions(inputs, recons, fig, gs[0])

        # plot slot attention mask
        if has_attn_mask:
            self.plot_masks(atten_masks, "slot attention masks", fig, gs[1])

        # plot decoder mask:
        if has_dec_mask:
            self.plot_masks(decoder_mask,"slot decoder masks", fig, gs[2])

        return self.name, fig

    def init_figure(self, n_slots, has_attn_mask, has_dec_mask):
        n_rows = self.H * (2 + n_slots * (has_attn_mask + has_dec_mask))
        n_rows += 1 + has_attn_mask + has_dec_mask
        n_cols = ((self.data_split == "both") + 1) * self.W * self.n_recons

        fig = plt.figure(layout="constrained", figsize=(n_cols, n_rows))

        grid_spec = fig.add_gridspec(n_rows, 1)

        sg_n_rows = 2 * self.H + 1
        gs_recons = grid_spec[:sg_n_rows].subgridspec(sg_n_rows, n_cols)

        if has_attn_mask:
            start_row, sg_n_rows = sg_n_rows, n_slots * self.H + 1
            gs_slot_masks = grid_spec[
                start_row:start_row + sg_n_rows
            ].subgridspec(sg_n_rows, n_cols)
        else:
            gs_slot_masks, start_row = None, 0

        if has_dec_mask:
            start_row += sg_n_rows
            sg_n_rows = n_slots * self.H + 1
            gs_dec_masks = grid_spec[
                start_row:start_row + sg_n_rows
            ].subgridspec(sg_n_rows, n_cols)
        else:
            gs_dec_masks = None

        return fig, (gs_recons, gs_slot_masks, gs_dec_masks)

    def plot_reconstructions(self, inputs, recons, figure, grid):
        if self.data_split == "both":
            titles = ["train", "test"]
        else:
            titles = [self.data_split]

        for i, t in enumerate(titles):
            title_width = self.W * self.n_recons
            self.add_section_title(
                figure, grid[0, i * title_width: (i + 1) * title_width], t
            )

        examples = np.stack([inputs.numpy(), recons.numpy()], axis=1)

        if examples.shape[-1] == 1:
            examples = np.repeat(examples, 3, -1)

        row_del = 1 + self.H
        for i, (input, recons) in enumerate(examples):
            c_start, c_end = self.W * i, self.W * (i + 1)

            ax_input = figure.add_subplot(grid[1: row_del, c_start: c_end])
            ax_recon = figure.add_subplot(grid[row_del:, c_start: c_end])

            ax_input.imshow(input)
            ax_recon.imshow(recons)

            strip_plot(ax_input)
            strip_plot(ax_recon)

            if i == 0:
                ax_input.set_ylabel("input", fontsize=20)
                ax_recon.set_ylabel("recons", fontsize=20)


    def plot_masks(self, masks, title, figure, grid):
        masks = masks.repeat(1, 1, 1, 1, 3).numpy()
        masks = colorize(masks)

        n_slots = masks.shape[1]
        labels = ['slot {}'.format(str(i+1)) for i in range(n_slots)]

        self.add_section_title(figure, grid[0, :], title)

        for i, example in enumerate(masks):
            for j, slot in enumerate(example):
                r_start, r_end = 1 + self.H * j, 1 + self.H * (j + 1)
                c_start, c_end = self.W * i, self.W * (i + 1)

                ax = figure.add_subplot(grid[r_start: r_end, c_start: c_end])

                ax.imshow(slot)

                strip_plot(ax)

                if i == 0:
                    ax.set_ylabel(labels[j], fontsize=20)

    def add_section_title(self, figure, grid_spec, title):
        ax = figure.add_subplot(grid_spec)
        ax.text(0.5, 0.5, title, fontsize=20, ha='center', va='center')
        strip_plot(ax)


class PerceptualGroupingLatentProjection(ReconstructionViz):
    def __init__(
        self,
        factor_dim: int,  # A factor we wish to plot in 3D. TODO: This could be a list
        comparison_factor_1: int,  # Factors to compare in 2D plot.
        comparison_factor_2: int,  # TODO: These two could be lists to plot several combinations
        dimensionality_reduction: Callable,  # this will be PSL but I am keeping it as a parameter
        n_proj_dims: int = 3,  # the number of dimnensios used for the representation
        simultaneous_projection: bool = True,
        subsample: int = 10000,
        name: str = "projection",
        load_computed_outputs: bool = False,
        save_folder: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.factor_dim = factor_dim
        self.comparison_factor_1 = comparison_factor_1
        self.comparison_factor_2 = comparison_factor_2
        self.n_proj_dims = n_proj_dims
        self.dim_reduction = dimensionality_reduction
        self.simultaneous_projection = simultaneous_projection
        self.subsample = subsample
        self.name = name
        self.load_computed_outputs = load_computed_outputs
        self.save_folder = save_folder
        # self.slot_selection_fn = slot_selection_fn

    def __call__(self, model, dataset):
        return self.create_figure(model, dataset)

    def create_figure(self, model, dataset):
        if self.load_computed_outputs and self.save_folder is not None:
           (
                train_embeddings,
                train_targets,
                test_embeddings,
                test_targets,
           ) = self.load_outputs()
        else:
            train_embeddings, train_targets = compute_embeddings(
                model, dataset.val_dataloader()
            )
            test_embeddings, test_targets = compute_embeddings(
                model, dataset.test_dataloader()
            )

        slot_idx = self.select_slot(model, train_embeddings, train_targets)

        train_embeddings = train_embeddings[:, slot_idx].numpy()
        test_embeddings = test_embeddings[:, slot_idx].numpy()
        train_targets = train_targets.numpy()
        test_targets = test_targets.numpy()

        self.fit_dim_reduction((train_embeddings, train_targets), (test_embeddings, test_targets))

        if self.subsample > 0:
            subsample_idx = np.random.choice(len(train_embeddings), size=self.subsample, replace=False)
            train_embeddings = train_embeddings[subsample_idx]
            train_targets = train_targets[subsample_idx]

            subsample_idx = np.random.choice(len(test_embeddings), size=self.subsample, replace=False)
            test_embeddings = test_embeddings[subsample_idx]
            test_targets = test_targets[subsample_idx]

        embedding_params_plot = self.plot_dim_reduction_params(dataset)
        latent_projection_plot = self.factor_projection(train_embeddings, test_embeddings)
        factor_comparison_plot = self.two_factor_comparison(
            dataset, train_embeddings, train_targets, test_embeddings, test_targets
        )

        return [
            ("weight_values", embedding_params_plot),
            ("latent_projection", latent_projection_plot),
            ("factor_comparison", factor_comparison_plot),
        ]

    def plot_dim_reduction_params(self, dataset):
        factor_labels = dataset.dataset_cls.factors

        # currently we are only interested in the y rotations, i.e. the weights that signal how
        # each dimension in the projection space contributes to representing the generative factor
        # space.
        y_rotations_ = self.dim_reduction.y_rotations_
        latent_contribution = np.abs(y_rotations_)  # shape (n_factors, n_dims)

        fig = plt.figure(figsize=(10, 10))  # should be n_factors == n_dims
        ax = fig.add_subplot()

        hinton(latent_contribution.T, factor_labels=factor_labels, ax=ax, fontsize=20)

        return  fig

    def factor_projection(
        self,
        train_manifold,
        test_manifold,
    ):
        y_rotations_ = self.dim_reduction.y_rotations_[self.factor_dim]
        index_dims = np.argsort(np.abs(y_rotations_))[-self.n_proj_dims:]

        train_manifold = train_manifold[:, index_dims]
        test_manifold = test_manifold[:, index_dims]

        # n_slots = train_manifold.shape[0]
        angles = list(range(0, 360, 60))
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10), subplot_kw={'projection':'3d'})

        # if not isinstance(axes, np.ndarray):
        #     axes = [axes]

        for ang, ax in zip(angles, axes.ravel()):
            ax.scatter(*train_manifold.T, color='k')
            ax.scatter(*test_manifold.T, color='r', marker='x')
            ax.view_init(azim=ang)

        return fig

    def two_factor_comparison(
        self,
        dataset,
        train_manifold,
        train_targets,
        test_manifold,
        test_targets
    ):
        train_manifold_proj, train_targets_proj = self.dim_reduction.transform(
            train_manifold, train_targets
        )

        test_manifold_proj, test_targets_proj = self.dim_reduction.transform(
            test_manifold, test_targets
        )

        y_rotations_ = self.dim_reduction.y_rotations_
        index_dims = np.argsort(np.abs(y_rotations_), axis=1)[
            [self.comparison_factor_1, self.comparison_factor_2], -1
        ]

        train_manifold_proj = train_manifold_proj[:, index_dims]
        train_targets_proj = train_targets_proj[:, index_dims]

        test_manifold_proj = test_manifold_proj[:, index_dims]
        test_targets_proj = test_targets_proj[:, index_dims]

        # train_predictions = self.dim_reduction.predict(train_manifold)
        # test_predictions = self.dim_reduction.predict(test_manifold)

        all_factors = dataset.dataset_cls.factors
        factors = all_factors[self.comparison_factor_1], all_factors[self.comparison_factor_2]

        # train_predictions = train_predictions[:, factor_idxs]
        # test_predictions = test_predictions[:, factor_idxs]
        # train_targets =  train_targets[:, factor_idxs]
        # test_targets =  test_targets[:, factor_idxs]

        train_scores = ((train_manifold_proj - train_targets_proj) ** 2).mean(axis=0)
        test_scores = ((test_targets_proj - test_targets_proj) ** 2).mean(axis=0)

        fig, (ax_pred, ax_scores) = plt.subplots(ncols=2, figsize=(20, 10))

        ax_pred.scatter(*train_manifold_proj.T, color='k')
        ax_pred.scatter(*test_manifold_proj.T, color='r', marker='x')

        ax_pred.set_xlabel(dataset.dataset_cls.factors[self.comparison_factor_1])
        ax_pred.set_ylabel(dataset.dataset_cls.factors[self.comparison_factor_2])

        bar_width = 0.25
        x = np.arange(len(factors))
        data_splits = ["train", "test"]

        for i, (split, scores) in enumerate(zip(data_splits, [train_scores, test_scores])):
            offset = bar_width * i
            rects = ax_scores.bar(x + offset, scores, bar_width, label=split)
            ax_scores.bar_label(rects, padding=3)

        ax_scores.set_xticks(x / 2 + bar_width, factors)
        ax_scores.legend(loc='upper left', ncols=3)

        return fig

    def fit_dim_reduction(self, train_data, test_data):
        train_slots, train_targets = train_data
        test_slots, test_targets = test_data

        if self.simultaneous_projection:
            all_embeddings = np.concatenate([train_slots, test_slots], axis=0)
            all_targets = np.concatenate([train_targets, test_targets], axis=0)
            self.dim_reduction.fit(all_embeddings, all_targets)
        else:
            self.dim_reduction.fit(train_slots, train_targets)

    def save_outputs(self, train_embeddings, train_targets, test_embeddings, test_targets):
        assert self.save_folder is not None
        file = osp.join(self.save_folder, "computed_projection.npz")
        np.savez(
            file,
            train_embeddings=train_embeddings,
            train_targets=train_targets,
            test_embeddings=test_embeddings,
            test_targets=test_targets
        )

    def load_outputs(self):
        assert self.save_folder is not None

        file = osp.join(self.save_folder, "computed_projection.npz")
        data_zip = np.load(file, allow_pickle=True)

        train_embeddings = data_zip['train_embeddings']
        test_embeddings = data_zip['test_embeddings']
        train_targets = data_zip['train_targets']
        test_targets = data_zip['test_targets']

        return train_embeddings, train_targets, test_embeddings, test_targets

    @torch.no_grad()
    def select_slot(self, model, inputs, targets):
        # currently assuming that this is a figure-ground segmentation model with only one slot
        return 0


class TraversalReconstruction(Visualzation):
    def __init__(
        self,
        traversal_dim: int,
        traversal_samples: int = 10,
        row_dim: Optional[int] = None,
        row_samples: int = 3,
        recons_transform: Literal["mse", "bce"] = "mse",
        recons_extractor: OutputSelector = IndexSelector(),
        include_inputs: bool = True,
        include_scores: bool = True,

    ):
        self.traversal_dim = traversal_dim
        self.traversal_samples = traversal_samples
        self.row_dim = row_dim
        self.row_samples = row_samples
        self.recons_transform = recons_transform
        self.recons_extractor = recons_extractor
        self.include_inputs = include_inputs
        self.include_scores = include_scores

    def create_figure(self, model, datamodule):
        examples, gf_values = self.get_examples(datamodule.full_dataloader())
        inputs, reconstructions = self.get_recons(model, examples)
        return self.plot(inputs, reconstructions, gf_values)

    def get_examples(self, loader):
        dataset = loader.dataset

        ga_values = np.stack([dataset.factor_values[i] for i in range(len(dataset))])
        unique_traversal_values = np.unique(ga_values[:, self.traversal_dim])
        unique_row_values = np.unique(ga_values[:, self.row_dim])

        inputs, targets = [], []
        input_size = dataset.img_size

        for trav_val in unique_traversal_values:
            for row_val in unique_row_values:
                idx = (
                    (ga_values[:, self.traversal_dim] == trav_val) &
                    (ga_values[:, self.row_dim] == row_val)
                )

                for i in range(ga_values.shape[1]):
                    if i != self.traversal_dim and i != self.row_dim:
                        unique_i = np.unique(ga_values[:, i])
                        idx = idx & (ga_values[:, i] == unique_i[len(unique_i) // 2])

                inputs.extend([dataset[i][0] for i in idx.nonzero()[0]])
                targets.extend(ga_values[idx])

        inputs = np.stack(inputs).reshape(
            len(unique_traversal_values), len(unique_row_values), *input_size,
        )
        targets = np.stack(targets).reshape(
            len(unique_traversal_values), len(unique_row_values), dataset.n_factors,
        )

        step = len(unique_traversal_values) // self.traversal_samples
        inputs = inputs[::step]
        targets = targets[::step]

        step = len(unique_row_values) // self.row_samples
        inputs = inputs[:,::step]
        targets = targets[:,::step]

        return inputs, targets

    @torch.no_grad()
    def get_recons(self, model, examples):
        inputs = torch.from_numpy(examples).flatten(0, 1)

        inputs = inputs.to(device=next(model.parameters()).device)
        outputs = model(inputs)

        recons = self.recons_extractor(outputs)

        if self.recons_transform == "mse":
            recons = recons.clamp(0, 1)
        else:
            recons = recons.sigmoid_()

        recons = recons.cpu().unflatten(0, examples.shape[:2])
        inputs = inputs.cpu().unflatten(0, examples.shape[:2])

        return inputs, recons

    def plot(self, examples, reconstructions, gf_values):
        T, R = examples.shape[:2]

        fig, axes = plt.subplots(2 * R, T, figsize=(1 * T, 2 * R))

        examples = examples.permute(0, 1, 3, 4, 2)
        reconstructions = reconstructions.permute(0, 1, 3, 4, 2)

        if examples.shape[-1] == 1:
            examples = np.repeat(examples, 3, -1)
        if reconstructions.shape[-1] == 1:
            reconstructions = np.repeat(reconstructions, 3, -1)

        for r in range(R):
            for t in range(T):
                # plt.imsave(
                #     f"/home/milton/Dropbox/IMAGE_SAMPLES/example_{r}_{t}.png",
                #     examples[t, r].numpy()
                # )

                # plt.imsave(
                #     f"/home/milton/Dropbox/IMAGE_SAMPLES/reconstructions_{r}_{t}.png",
                #     reconstructions[t, r].numpy()
                # )

                ax_ex = axes[2 * r, t]
                ax_rec = axes[2 * r + 1, t]

                ax_ex.imshow(examples[t, r])
                ax_rec.imshow(reconstructions[t, r])

                strip_plot(ax_ex)
                strip_plot(ax_rec)

                if t == 0:
                    ax_ex.set_ylabel("input", fontsize=12)
                    ax_rec.set_ylabel("recons", fontsize=12)

                # if r == (R - 1):
                #     ax_rec.set_xlabel(gf_values[t, r, self.traversal_dim])

        plt.tight_layout()

        return "traversal_recons", fig


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


def strip_plot(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.set_xlabel('')
    ax.set_ylabel('')

    ax.set_xticks([])
    ax.set_yticks([])
