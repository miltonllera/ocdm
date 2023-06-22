import random
from typing import Any, Dict, List, Optional, Type

import torch
import torch.nn as nn
from pytorch_lightning.core.module import LightningModule
from pytorch_lightning.core.datamodule import LightningDataModule
from torch.utils.data import Dataset, DataLoader, random_split

from bin.init.config import load_model
from .ood_loader import OODLoader
from .sampler import ImbalancedSampler


class AdversarialDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        model: LightningModule,
    ):
        self.dataset = dataset
        self.model = model.eval()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x = self.dataset[index][0]

        if random.random() < 0.5:

            with torch.no_grad():
                x = self.model(x[None])[0][0]

            y = torch.tensor([0], dtype=torch.float32)
        else:
            y = torch.tensor([1], dtype=torch.float32)

        return x, y


class AdversarialDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_cls: Type[Dataset],
        path: str,
        train_generative_model_path: str,
        test_generative_model_path: str,
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

        self.train_model = load_model(train_generative_model_path)
        self.test_model = load_model(test_generative_model_path)

        self.rebalance_wrt_factor = rebalance_wrt_factor

    def setup(self, stage: Optional[str] = None) -> None:
        if stage != "fit" and hasattr(self, "test_data"):
            return

        dataset = self.loader.load_dataset('test')

        if stage == 'fit':
            dataset = AdversarialDataset( dataset, self.train_model)
            self.train_data, self.val_data = random_split(dataset, [0.9, 0.1])
        else:
            self.test_data = AdversarialDataset(dataset, self.test_model)

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
            self.val_data,
            self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
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

    def get_sampler(self, dataset):
        if self.rebalance_wrt_factor is not None:
            factors = dataset.factors
            factor = self.rebalance_wrt_factor
            cat_idx = [i for i, s in enumerate(factors) if factor in s][0]
            labels = dataset.factor_values[:,cat_idx]
            return ImbalancedSampler(labels)

        return None
