import torch
import pandas as pd
from torch.utils.data.sampler import Sampler


class ImbalancedSampler(Sampler):
    """
    Rebalances a dataset so that all labels are presented the same amount of times.

    Based on code found (here)[https://github.com/ufoym/imbalanced-dataset-sampler].
    """
    def __init__(self, labels) -> None:
        self.indices = list(range(len(labels)))
        self.num_samples = len(self.indices)

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] =  labels
        df.index = pd.Index(self.indices)
        df.sort_index(inplace=True)

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True
        ))

    def __len__(self):
        return self.num_samples
