"""
Sprite dataset based on the Pentomino shapes.

The advantage of this dataset is that it has more shapes than similar ones such
as dSprites (12 vs 3) and the shapes share features that only vary in their
spatial configuration (they are all composed of squares).

More details can be found at:

    https://en.wikipedia.org/wiki/Pentomino

"""

import os.path as osp
import json
from itertools import product
from typing import Callable, Optional

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as trans


class Pentominos(Dataset):
    n_factors = 6
    factors = ('shape', 'color', 'scale', 'angle', 'pos_x', 'pos_y')

    def __init__(
        self,
        path: str,
        prediction_type: str = 'unsupervised',
        held_out_filter: Callable = None,
    ) -> None:
        (
            image_files,
            factor_values,
            factor_classes
        ) = self.load_raw(path, held_out_filter)

        self.image_files = image_files
        self.factor_values = factor_values
        self.factor_classes = factor_classes
        self.transform = trans.ToTensor()
        self.prediction_type = prediction_type

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image = self.load_image(self.image_files[index])
        if self.prediction_type == 'unsupervised':
            return image, image
        elif self.prediction_type == 'classification':
            return image, self.factor_classes[index].astype(np.int64)
        return image, self.factor_values[index]

    def load_image(self, path):
        image = self.transform(Image.open(path))

        if len(self.unique_values['colors']) == 1:  # black and white image
            image = image[:1]

        return image

    @staticmethod
    def load_raw(path, factor_filter=None):
        data_file = osp.join(path, "pentominos.json")
        with open(data_file) as f:
            data_info = json.load(f)
            data = data_info['data']
            meta = data_info['meta']

        image_files, factor_values = [], []
        for example_info in data:
            image_path = osp.join(
                path, "images", example_info['image_file_name']
            )

            image_files.append(image_path)
            factor_values.append([example_info[f] for f in Pentominos.factors])

        n_values = tuple(len(fv) for fv in meta['unique_values'].values())

        # Set meta values
        img_size = (
            3 if (len(meta['unique_values']['colors']) > 1) else 1,
            meta['height'],
            meta['width']
        )

        factor_values = np.array(factor_values, dtype=np.float32)
        factor_classes = list(product(*[range(i) for i in n_values]))

        if n_values[-1] > 1 and n_values[1] > 1:  # more than one background color
            factor_classes = [c for c in factor_classes if c[-1] != c[1]]
        factor_classes = np.asarray(factor_classes, dtype=np.float32)

        # This is a complete anti-pattern that should not be used ever
        Pentominos.unique_values = meta['unique_values']
        Pentominos.img_size = img_size
        Pentominos.factor_sizes = n_values
        Pentominos.shape_names = np.asarray([s for s in meta['shape_names'].values()])

        # Remove excluded values
        if factor_filter is not None:
            idx = factor_filter(factor_values)

            image_files = [image_files[i] for i in idx.nonzero()[0]]
            factor_values = factor_values[idx]
            factor_classes = factor_classes[idx]

            if len(image_files) == 0:
                raise ValueError("Condition filter removed all data")

        assert len(image_files) == len(factor_values) == len(factor_classes)

        return image_files, factor_values, factor_classes

    def map_shapes(self, values):
        return type(self).shape_names[values.astype(int)]


class FixedRotationPentominos(Pentominos):
    def __init__(
        self,
        path: str,
        factor_filter: Optional[Callable],
    ) -> None:
        (
            target_images,
            target_fvs,
            target_classes
        ) = self.load_raw(path)

        self.target_image_files = target_images
        self.target_factor_values = target_fvs
        self.target_classes = target_classes

        total_combs = np.prod(self.factor_sizes)
        self.code_bases = total_combs / np.cumprod(self.factor_sizes)

        super().__init__(path, factor_filter)

    def __getitem__(self, index):
        image, _, input_classes = super().__getitem__(index)
        target_fv = input_classes.copy()
        # since rotation values are 9 degrees apart, 5 values is 45 degrees
        target_fv[3] = (target_fv[3] + 5) % self.factor_sizes[3]

        idx = int(np.dot(target_fv, self.code_bases))
        rotated_image = self.load_image(self.target_image_files[idx])

        return image, rotated_image


class Pentominos3D(Dataset):
    n_factors = 6
    factors = (
        'object', 'rotation_y', 'color_hue', 'wall_color_hue', 'floor_color_hue', 'camera_angle'
    )

    def __init__(
        self,
        path: str,
        prediction_type: str = 'unsupervised',
        held_out_filter: Callable | None = None,
    ) -> None:
        meta_path = osp.join(path, 'metadata.csv')
        df = pd.read_csv(meta_path)
        df.columns = df.columns.str.strip()

        shape_names = sorted(df['object'].unique())
        shape_to_idx = {s: i for i, s in enumerate(shape_names)}

        image_files = [osp.join(path, fn) for fn in df['filename']]
        factor_cols = [
            'object', 'rotation_y', 'color_hue', 'Wall_color_hue', 'Floor_color_hue', 'camera_angle'
        ]
        raw_values = df[factor_cols].copy()
        raw_values['object'] = raw_values['object'].map(shape_to_idx)
        factor_values = raw_values.to_numpy(dtype=np.float32)

        unique_per_factor = [np.sort(np.unique(factor_values[:, i])) for i in range(self.n_factors)]
        factor_classes = np.stack(
            [np.searchsorted(unique_per_factor[i], factor_values[:, i]) for i in range(self.n_factors)],
            axis=1,
        ).astype(np.float32)

        Pentominos3D.img_size = (3, 128, 128)
        Pentominos3D.factor_sizes = tuple(len(u) for u in unique_per_factor)
        Pentominos3D.shape_names = np.asarray(shape_names)
        Pentominos3D.unique_values = {f: unique_per_factor[i].tolist() for i, f in enumerate(self.factors)}

        if held_out_filter is not None:
            idx = held_out_filter(factor_values)
            image_files = [image_files[i] for i in idx.nonzero()[0]]
            factor_values = factor_values[idx]
            factor_classes = factor_classes[idx]
            if len(image_files) == 0:
                raise ValueError("Condition filter removed all data")

        self.image_files = image_files
        self.factor_values = factor_values
        self.factor_classes = factor_classes
        self.transform = trans.Compose([trans.Resize((128, 128)), trans.ToTensor()])
        self.prediction_type = prediction_type

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image = self.load_image(self.image_files[index])
        if self.prediction_type == 'unsupervised':
            return image, image
        elif self.prediction_type == 'classification':
            return image, self.factor_classes[index].astype(np.int64)
        return image, self.factor_values[index]

    def load_image(self, path):
        return self.transform(Image.open(path).convert('RGB'))

    def map_shapes(self, values):
        return type(self).shape_names[values.astype(int)]


def shape_prediction(targets: np.ndarray) -> np.ndarray:
    object_shape = np.zeros(12, dtype=np.float32)
    object_shape[int(targets[0])] = 1.  # one hot shape
    return object_shape[None]


def rotation_prediction(targets: np.ndarray) -> np.ndarray:
    object_rotation = np.zeros(1, dtype=np.float32)
    object_rotation[0] = targets[3]
    return object_rotation[None]

