import torch
import polars as pl
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
        df = pl.DataFrame({ "index": self.indices, "label": labels }).sort("index")

        # Count occurrences of each label
        label_counts = df.group_by("label").agg(pl.count().alias("count"))

        # Join counts back to the original dataframe
        df = df.join(label_counts, on="label")

        # Calculate weights as inverse of counts
        weights = 1.0 / df["count"]

        self.weights = torch.DoubleTensor(weights.to_list())

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True
        ))

    def __len__(self):
        return self.num_samples
