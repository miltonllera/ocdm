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
from .comptask import IndexMap



class FixedPropertyCommandTask(DatasetWrapper):
    def __init__(self, base_dataset, generative_factor_index):
        super().__init__(base_dataset)
        self.index_map = IndexMap(base_dataset)
        self.generative_factor_index = generative_factor_index

    def __getitem__(self, idx):
        gt_values = self.factor_values[idx]

        transformed_gt_values = gt_values.copy()
