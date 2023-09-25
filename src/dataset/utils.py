import numpy as np
from torch.utils.data.dataset import Dataset


class class_property:
    def __init__(self, getter) -> None:
        super().__init__()
        self.getter = getter

    def __get__(self, instance, owner):
        return self.getter(owner)


class Supervised:
    def __init__(self, pred_type='reg', dim=None, target_transform=None):
        self.pred_type = pred_type
        self.dim = dim
        self.target_transform = target_transform

    def __call__(self, image, factor_values, factor_classes):
        if self.pred_type == "class":
            target = factor_classes.astype(np.int32)
        else:
            target = factor_values.astype(np.float32)

        if self.dim is not None:
            target = target[self.dim]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target


class Unsupervised:
    def __call__(self, image, factor_values, factor_classes):
        return image, image


class DatasetWrapper(Dataset):
    def __init__(self, base_dataset):
        self.dataset = base_dataset

    def __len__(self):
        return len(self.dataset)
        # return self.n_samples

    @property
    def n_factors(self):
        return len(self.factor_sizes)

    @property
    def factor_sizes(self):
        return self.dataset.factor_sizes

    @property
    def img_size(self):
        return self.dataset.img_size

    @property
    def factors(self):
        return self.dataset.factors

    @property
    def factor_code(self):
        return self.dataset.factor_classes

    @property
    def factor_values(self):
        return self.dataset.factor_values

    @property
    def transform(self):
        return self.dataset.transform

    @property
    def categorical(self):
        return self.dataset.categorical
