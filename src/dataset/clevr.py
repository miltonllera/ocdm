"""
CLEVR dataset.

This is the compositional generalization version of the dataset which has different combinations
of colors and shapes for training and testing.
"""

import os.path as osp
import json
from PIL import Image
from typing import Callable

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.transforms import  ToTensor

from torchvision.transforms import Compose, ToTensor, CenterCrop, Resize


class CLEVR(Dataset):
    def __init__(self, dataset_name, root, condition, split, setting) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.root = root
        self.condition = condition
        self.split = split
        self.setting = setting

        if self.setting == "qa":
            with open(self.qa_file) as qa_file:
                self.qa = json.load(qa_file)
        else:
            with open(self.scene_file) as scene_file:
                self.scenes = self.filter_scenes(json.load(scene_file)['scenes'])

        self.transform: Callable[[Image.Image], torch.Tensor] = Compose([
            ToTensor(),
            CenterCrop((192, 192)),
            Resize((64, 64)),
        ])

    def __len__(self):
        if hasattr(self, "scenes"):
            return len(self.scenes)
        else:
            return len(self.qa)

    def __getitem__(self, index):
        if self.setting == "unsupervised":
            image_file = self.scenes[index]["image_filename"]
            image = self.load_image(image_file)[:-1]  # exclude fine
            return image, image
        else:
            raise NotImplementedError()

    @property
    def image_folder(self):
        return osp.join(self.root, "images", f"{self.split}{self.condition}")

    @property
    def qa_file(self):
        return osp.join(
            self.root,
            "questions",
            f"CLEVR_{self.split}{self.condition}_questions.json"
        )

    @property
    def scene_file(self):
        return osp.join(
            self.root,
            "scenes",
            f"CLEVR_{self.split}{self.condition}_scenes.json"
        )

    def load_image(self, filename):
        full_path = osp.join(self.image_folder, filename)
        with Image.open(full_path) as pil_image:
            image = self.transform(pil_image)
        return image

    def filter_scenes(self, scenes):
        if self.dataset_name == "clevr6":
            return list(filter(lambda x: len(x['objects']) < 7, scenes))
        if self.dataset_name == "clevr10":
            return list(filter(lambda x: len(x['objects']) < 11, scenes))
        return scenes


class CLEVRDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_name,
        root,
        condition,
        setting,
        batch_size: int = 64,
        eval_batch_size: int = 1,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.root = root
        self.condition = condition
        self.setting = setting
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers

    def setup(self, stage: str) -> None:
        if stage in ["test", "predict"] and self.condition == "interp":
            raise NotImplementedError()
            # self.test_data = CLEVR(
            #     self.dataset_name,
            #     self.root,
            #     "A",
            #     "test",
            #     self.setting,
            # )
        elif stage in ["test", "predict"] and self.condition == "ood":
            self.test_data = CLEVR(
                self.dataset_name,
                self.root,
                "B",
                "val",
                self.setting,
            )
        else:
            self.train_data = CLEVR(
                self.dataset_name,
                self.root,
                "A",
                "train",
                self.setting
            )
            self.val_data = CLEVR(
                self.dataset_name,
                self.root,
                "A",
                "val",
                self.setting
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_data,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_data,
            self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
