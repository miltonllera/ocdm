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
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as trans

from .utils import class_property


class Pentominos(Dataset):
    n_factors = 6

    factors = ('shape', 'color', 'scale', 'angle', 'pos_x', 'pos_y')

    def __init__(
        self,
        path: str,
        factor_filter: Optional[Callable],
        _getter: Optional[Callable] = None,
    ) -> None:
        (
            image_files,
            factor_values,
            factor_classes
        ) = self.load_raw(path, factor_filter)

        self.image_files = image_files
        self.factor_values = factor_values
        self.factor_classes = factor_classes
        self.transform = trans.ToTensor()
        self._getter = _getter

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image = self.load_image(self.image_files[index])
        data = image, self.factor_values[index], self.factor_classes[index]

        if self._getter is None:
            return data
        return self._getter(*data)

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

        factor_values = np.array(factor_values, dtype=np.float32)
        factor_classes = np.asarray(
            list(product(*[range(i) for i in n_values])), dtype=np.int32
        )

        # Set meta values
        img_size = (
            3 if (len(meta['unique_values']['colors']) > 1) else 1,
            meta['height'],
            meta['width']
        )

        # This is a complete anti-pattern that should not be used ever
        Pentominos.unique_values = meta['unique_values']
        Pentominos.img_size = img_size
        Pentominos.factor_sizes = n_values

        # Remove excluded values
        if factor_filter is not None:
            idx = factor_filter(factor_values, factor_classes)

            image_files = [image_files[i] for i in idx.nonzero()[0]]
            factor_values = factor_values[idx]
            factor_classes = factor_classes[idx]

            print(len(image_files))

            if len(image_files) == 0:
                raise ValueError("Condition filter removed all data")

        assert len(image_files) == len(factor_values) == len(factor_classes)

        return image_files, factor_values, factor_classes


    @staticmethod
    def get_splits():
        return {
            'combgen': {
                'shape_and_rotation': _masks.shape_and_rotation,
                'scale_combgen_from_loo': _masks.scale_combgen_from_loo,
            },
            'extrap': {
                'new_shape': _masks.new_shape,
                'three_new_shapes': _masks.three_new_shapes,
                'half_new_shapes': _masks.half_new_shapes,
                'rotated_cannonical_shape': _masks.rotated_cannonical_shape,
            }
        }

    @staticmethod
    def get_modifiers():
        return {
            'remove_redundant_rotations': _masks.remove_redundant_rotations,
        }


class FixedRotationPentominos(Pentominos):
    def __init__(
        self,
        path: str,
        factor_filter: Optional[Callable],
        _getter: Optional[Callable] = None
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

        super().__init__(path, factor_filter,  _getter=None)

    def __getitem__(self, index):
        image, _, input_classes = super().__getitem__(index)

        target_fv = input_classes.copy()
        # since rotation values are 9 degrees apart, 5 values is 45 degrees
        target_fv[3] = (target_fv[3] + 5) % self.factor_sizes[3]

        idx = int(np.dot(target_fv, self.code_bases))
        rotated_image = self.load_image(self.target_image_files[idx])

        return image, rotated_image


def shape_prediction(targets: np.ndarray) -> np.ndarray:
    object_shape = np.zeros(12, dtype=np.float32)
    object_shape[int(targets[0])] = 1.  # one hot shape
    return object_shape[None]


def rotation_prediction(targets: np.ndarray) -> np.ndarray:
    object_rotation = np.zeros(1, dtype=np.float32)
    object_rotation[0] = targets[3]
    return object_rotation[None]


class _masks:
    shp, hue, scl, rot, tx, ty = 0, 1, 2, 3, 4, 5

    @class_property
    def remove_redundant_rotations(cls):
        def modifier(factor_values, factor_classes):
            i_rotations = (
                (factor_values[:, cls.shp] == 0) &
                (factor_values[:, cls.rot] < 180)
            )

            x_rotations = (
                (factor_values[:, cls.shp] == 9) &
                (factor_values[:, cls.rot] < 90)
            )

            z_rotations = (
                (factor_values[:, cls.shp] == 11) &
                (factor_values[:, cls.rot] < 180)
            )

            rest = ~np.isin(factor_values[:, cls.shp], [0, 9, 11])

            return i_rotations | z_rotations | x_rotations | rest

        return modifier

    @class_property
    def shape_and_rotation(cls):
        def test_mask(factor_values, factor_classes):
            excluded_shapes = np.isin(factor_values[:, cls.shp], [1, 3, 5, 8])
            excluded_angles = factor_values[:, cls.rot] < 180
            return excluded_shapes & excluded_angles

        def train_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return train_mask, test_mask

    @class_property
    def new_shape(cls):
        def test_mask(factor_values, factor_classes):
            return factor_values[:, cls.shp] == 8

        def train_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return train_mask, test_mask

    @class_property
    def three_new_shapes(cls):
        def test_mask(factor_values, factor_classes):
            return np.isin(factor_values[:, cls.shp], [3, 5, 8])

        def train_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return train_mask, test_mask

    @class_property
    def half_new_shapes(cls):
        def test_mask(factor_values, factor_classes):
            return np.isin(factor_values[:, cls.shp], [1, 3, 4, 5, 7, 8])

        def train_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return train_mask, test_mask

    @class_property
    def scale_combgen_from_loo(cls):
        def test_mask(factor_values, factor_classes):
            return (
                (factor_classes[:, cls.scl] == 4) &  # exclude large scales
                (factor_values[:, cls.shp] != 8)     # except for all but one shape
            )

        def train_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return train_mask, test_mask

    @class_property
    def rotated_cannonical_shape(cls):
        def test_mask(factor_values, factor_classes):
            return (
                (factor_values[:, cls.shp] == 8) &
                (factor_values[:, cls.scl] == 3) &
                (factor_classes[:, cls.tx] == 10) &
                (factor_classes[:, cls.ty] == 10)
            )

        def train_mask(factor_values, factor_classes):
            return (
                (factor_values[:, cls.shp] == 1) &
                (factor_values[:, cls.scl] == 3) &
                (factor_classes[:, cls.tx] == 10) &
                (factor_classes[:, cls.ty] == 10)
            )

        return train_mask, test_mask
