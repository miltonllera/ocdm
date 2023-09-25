from typing import Any, Dict, List,  Optional, Type

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, Dataset

from .utils import DatasetWrapper
from .ood_loader import OODLoader
from .sampler import ImbalancedSampler


class CompositionTaskDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_cls: Type[Dataset],
        path: str,
        split_condition: Optional[str] = None,
        split_variant: Optional[str] = None,
        split_modifiers: Optional[List[str]] = None,
        transform: Optional[nn.Module] = None,
        batch_size: int = 64,
        num_workers: int = 4,
        rebalance_wrt_factor: Optional[int] = None,
        _dataset_params: Optional[Dict[str, Any]] = None,
    ) -> None:

        if _dataset_params is None:
            _dataset_params = {}

        super().__init__()
        self.tranform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.loader = OODLoader(
            dataset_cls,
            path,
            split_condition,
            split_variant,
            split_modifiers,
            _dataset_params
        )

        self.rebalance_wrt_factor = rebalance_wrt_factor

    def setup(self, stage: Optional[str] = None) -> None:
        if stage != "fit" and hasattr(self, "test_data"):
            return

        dataset = CompositionTask(self.loader.load_dataset(stage))

        if stage == "fit" and self.loader.split_train is not None:
            self.train_data = dataset
        elif stage != "fit" and self.loader.split_test is not None:
            self.test_data = dataset
        elif self.loader.split_test is None:
            self.train_data = self.test_data = dataset

    def train_dataloader(self) -> DataLoader:
        sampler = self.get_sampler(self.train_data)
        return DataLoader(
            self.train_data,
            self.batch_size,
            num_workers=self.num_workers,
            shuffle=sampler is None,
            pin_memory=True,
            sampler=sampler,
        )

    def val_dataloader(self) -> DataLoader:
        sampler = self.get_sampler(self.train_data)
        return DataLoader(
            self.train_data,
            self.batch_size,
            num_workers=self.num_workers,
            shuffle=sampler is None,
            pin_memory=True,
            sampler=sampler,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_data,
            self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,  # we use limit_test_batches, so shuffle
            pin_memory=True,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def get_sampler(self, dataset):
        if self.rebalance_wrt_factor is not None:
            factors = dataset.factors
            factor = self.rebalance_wrt_factor
            cat_idx = [i for i, s in enumerate(factors) if factor in s][0]
            labels = dataset.factor_values[:,cat_idx]
            return ImbalancedSampler(labels)

        return None


class CompositionTask(DatasetWrapper):
    def __init__(self, base_dataset):
        super().__init__(base_dataset)
        self.index_map = IndexMap(base_dataset)

    def __getitem__(self, idx):
        z_code = self.factor_code[idx]  # type: ignore
        transf_z_code = z_code.copy()

        # select factor to transform and value to transform to
        all_factors = np.arange(self.n_factors)  # type: ignore
        np.random.shuffle(all_factors)

        # Iterate through all dimensions until we sample a new value
        dim = None
        for dim in all_factors:
            new_dim_code = self.sample_factor(z_code, dim)

            # Only assign if we could sample a new value for that dimension
            # if it's the last dimension, then we don't have any other choice
            if (new_dim_code != transf_z_code[dim]) or (dim == all_factors[-1]):
                transf_z_code[dim] = new_dim_code
                break

        # sample a command image
        command_z_code = transf_z_code.copy()
        for d in range(len(self.factor_sizes)):  # type: ignore
            if d != dim:
                command_z_code[d] = self.sample_factor(command_z_code, d)

        action = one_hot(
            torch.LongTensor([dim]),
            num_classes=self.n_factors  # type: ignore
        ).squeeze()

        img = self.dataset[idx][0]
        command_img = self.dataset[self.code_to_image(command_z_code)][0]
        transformed_img = self.dataset[self.code_to_image(transf_z_code)][0]

        input_imgs = torch.stack([img, command_img], dim=0).contiguous()
        target = torch.stack(
            [img, command_img, transformed_img],
            dim=0,
        ).contiguous()

        return (input_imgs, action), target

    def sample_factor(self, factor_code, dim):
        factor_d_code = np.arange(self.factor_sizes[dim])

        # Determine which codes are valid
        possible_codes = np.repeat(
            factor_code[None],
            torch.asarray([len(factor_d_code)]),
            axis=0,
        )
        possible_codes[:, dim] = factor_d_code

        idxs = self.code_to_index(possible_codes)
        is_valid = self.index_map[idxs] != -1

        # If more than one value is valid, remove the current one
        if sum(is_valid) > 1:
            is_valid[factor_code[dim]] = False
        # else return current
        else:
            return factor_code[dim]

        prob = np.ones(self.factor_sizes[dim]) / (sum(is_valid))
        prob[~is_valid] = 0

        return np.random.choice(factor_d_code, p=prob)

    def code_to_index(self, code):
        return self.index_map.index(code)

    def code_to_image(self, code):
        idx = self.index_map[self.code_to_index(code)]
        return self.dataset[idx][0]


class IndexMap:
    """
    Index map that for a given index in the full dataset, returns the corresponding
    index after applying a filter that excludes some factor combinations.

    Use this to sample relevant combinations the composition task.
    """
    def __init__(self, dataset):
        total_combs = np.prod(dataset.factor_sizes)

        self.code_bases = total_combs / np.cumprod(dataset.factor_sizes)

        if total_combs == len(dataset):
            index_table = None
        else:
            index_table = np.zeros(np.prod(dataset.factor_sizes),
                                   dtype=np.int64) - 1

            for i, c in enumerate(dataset.factor_classes):
                index_table[self.index(c)] = i

        self.index_table = index_table

    def __getitem__(self, item):
        if self.index_table is None:
            return item
        return self.index_table[item]

    def index(self, code):
        return np.asarray(np.dot(code, self.code_bases), np.int64)

    def is_valid(self, code):
        return self[self.index(code)] != -1
