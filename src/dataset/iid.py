from typing import Any, Dict, List, Literal, Optional, Sequence, Union

# import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning.core.datamodule import LightningDataModule
from torch.utils.data import random_split, DataLoader, Subset

from .ood_loader import OODLoader
from .utils import Supervised, Unsupervised
from .sampler import ImbalancedSampler


class IIDDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        path: str,
        split_condition: Optional[str] = None,
        split_variant: Optional[str] = None,
        split_modifiers: Optional[List[str]] = None,
        setting: Optional[Literal["unsupervised", "supervised"]] = None,
        transform: Optional[nn.Module] = None,
        batch_size: int = 64,
        num_workers: int = 4,
        split_data: bool = True,
        split_seed: Optional[int] = None,  # useful for distributed training
        split_sizes: Sequence[Union[int, float]] = [0.90, 0.05, 0.05],
        rebalance_wrt_factor: Optional[int] = None,
        _dataset_params: Optional[Dict[str, Any]] = None,
        _getter_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        if _dataset_params is None:
            _dataset_params = {}

        if _getter_params is None:
            _getter_params = {}

        super().__init__()
        self.tranform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_data = split_data
        self.split_seed = split_seed
        self.split_sizes = split_sizes

        if setting == "supervised":
            getter = Supervised(**_getter_params)
        elif setting == "unsupervised":
            getter = Unsupervised()
        else:
            getter = None

        _dataset_params['_getter'] = getter

        self.loader = OODLoader(
            dataset_name,
            path,
            split_condition,
            split_variant,
            split_modifiers,
            _dataset_params
        )

        self.rebalance_wrt_factor = rebalance_wrt_factor

        assert (not split_data or
            (len(split_sizes) == 3 and self.loader.split_test is None) or
            (len(split_sizes)== 2 and self.loader.split_test is not None)
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage != "fit" and hasattr(self, "test_data"):
            return

        dataset = self.loader.load_dataset(stage)

        # mean_values, min_values, max_values = [], [], []
        # for f in dataset.factors:
        #     mean_values.append(dataset.unique_values[f].mean())
        #     min_values.append(dataset.unique_values[f].min())
        #     max_values.append(dataset.unique_values[f].max())

        # print(min_values)
        # print(max_values)
        # print(mean_values)

        # exit()

        if stage in ["test", "predict"]:
             self.test_data = dataset

        elif not self.split_data:
            self.train_data = self.val_data = dataset

        else:
            if self.split_seed is not None:
                generator = torch.Generator().manual_seed(self.split_seed)
            else:
                generator = None

            splits = random_split(
                dataset=dataset,
                lengths=self.split_sizes,
                generator=generator
            )

            self.train_data, self.val_data = splits[:2]

            if len(splits) == 3:
                self.test_data = splits[2]


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
        return DataLoader(
            self.val_data,
            self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            sampler=self.get_sampler(self.val_data),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_data,
            1,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def get_sampler(self, dataset):
        if self.rebalance_wrt_factor is None:
            return None

        factor_name = self.rebalance_wrt_factor

        if isinstance(dataset, Subset):
            factors = dataset.dataset.factors
            values = dataset.dataset.factor_values
        else:
            factors = dataset.factors
            values = dataset.factor_values

        cat_idx = factors.index(factor_name)
        labels = values[:,cat_idx]

        if isinstance(dataset, Subset):
            labels = labels[dataset.indices]

        return ImbalancedSampler(labels)
