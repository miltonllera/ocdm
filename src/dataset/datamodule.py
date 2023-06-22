import logging
import torch
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from typing import Callable, Literal

from .comptask import CompositionTask
from .utils import build_filter
from .sampler import ImbalancedSampler


_logger = logging.getLogger(__name__)


class DisentangledDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: Callable[[str, str, Callable], Dataset],
        prediction_type: Literal['unsupervised', 'classification', 'regression'] = 'unsupervised',
        held_out_filter: Callable | str | None = None,
        batch_size: int = 64,
        num_workers: int = 4,
        val_split: float = 0.1,
        test_split: float = 0.1,
        rebalance_dim: int | str | None = None,
        seed: int = 42,
        path: str | None = None,
    ):
        if isinstance(held_out_filter, str):
            held_out_filter = build_filter(dataset, held_out_filter)
        assert held_out_filter is None or callable(held_out_filter)

        # Validate splits sum to 1
        if held_out_filter is not None:
            test_split = 0.0

        super().__init__()
        self.dataset = dataset
        self.held_out_filter = held_out_filter
        self.prediction_type = prediction_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_ratio = val_split
        self.test_ratio = test_split
        self.rebalance_dim = rebalance_dim
        self.seed = seed
        self.dataset_path = path
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str | None = None):
        """Setup train/val/test splits."""
        if (
            stage == 'fit' and self.train_dataset is not None or
            stage == 'test' and self.test_dataset is not None
        ):
            return self

        if self.held_out_filter is not None and stage == 'fit':
            data_filter = lambda *args: ~self.held_out_filter(*args)  # type: ignore
        else:
            data_filter = self.held_out_filter

        full_dataset = self.dataset(self.dataset_path, self.prediction_type, data_filter)

        # Calculate split sizes
        if stage == 'fit':
            total_size = len(full_dataset)
            val_size = int(self.val_ratio * total_size)
            test_size = int(self.test_ratio * total_size) if self.held_out_filter is None else 0
            train_size = total_size - val_size - test_size

            if test_size == 0 and self.held_out_filter is None:
                _logger.warning("Test split size is 0, but no held_out_filter value was provided")

            generator = torch.Generator().manual_seed(self.seed)
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                full_dataset,
                [train_size, val_size, test_size],
                generator=generator
            )
            # NOTE: Set the test_dataset to None if it is empty, otherwise leave it so we can
            # return immediately when calling setup again in line 54
            if len(self.test_dataset) == 0:
                self.test_dataset = None
        else:
            # NOTE: Because of the check in line 85, if no held_out_filter was provided the
            # test_dataset will already be set and returned in line 58 using the check in line 54
            self.test_dataset = full_dataset

        return self

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None
        sampler = self.get_sampler(self.train_dataset)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=sampler,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset is not None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_sampler(self, dataset):
        if (factor_name := self.rebalance_dim) is None:
            return None

        if isinstance(dataset, Subset):
            factors = dataset.dataset.factors  # type: ignore
            values = dataset.dataset.factor_values  # type: ignore
        else:
            factors = dataset.factors
            values = dataset.factor_values

        cat_idx = factors.index(factor_name) if isinstance(factor_name, str) else factor_name
        labels = values[:,cat_idx]

        if isinstance(dataset, Subset):
            labels = labels[dataset.indices]

        return ImbalancedSampler(labels)


class CompositionTaskDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: Callable[[str, str, Callable], Dataset],
        held_out_filter: Callable | str | None = None,
        batch_size: int = 64,
        num_workers: int = 4,
        rebalance_dim: int | str | None = None,
        seed: int | None = None,
        path: str | None = None,
    ) -> None:

        super().__init__()
        self.dataset = dataset
        self.held_out_filter = held_out_filter
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.rebalance_dim = rebalance_dim
        self.seed = seed
        self.dataset_path = path
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str | None = None):
        """Setup train/val/test splits."""
        if (
            stage == 'fit' and self.train_dataset is not None or
            stage == 'test' and self.test_dataset is not None
        ):
            return

        dataset = self.dataset(self.dataset_path, "unsupervised", self.held_out_filter)
        if stage == "fit":
            # NOTE: we only use the composition formulation for training.
            self.train_dataset = CompositionTask(dataset, self.seed)
            self.val_dataset = dataset
        else:
            self.test_dataset = dataset

    def train_dataloader(self) -> DataLoader:
        sampler = self.get_sampler(self.train_dataset)
        return DataLoader(
            self.train_dataset,
            self.batch_size,
            num_workers=self.num_workers,
            shuffle=sampler is None,
            pin_memory=True,
            sampler=sampler,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def get_sampler(self, dataset):
        dataset = dataset.dataset

        if self.rebalance_dim is not None:
            factors = dataset.factors
            factor = self.rebalance_dim
            cat_idx = [i for i, s in enumerate(factors) if factor in s][0]
            labels = dataset.factor_values[:,cat_idx]
            return ImbalancedSampler(labels)

        return None

