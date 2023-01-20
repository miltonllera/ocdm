"""
dSprites dataset module

The module contains the code for loading the dSprites dataset. This dataset
contains transformations of simple sprites in 2 dimensions, which have no
detailed features.

The original dataset can be found at:
    https://github.com/deepmind/3d-shapes
"""


import h5py
from typing import Optional, Callable

import numpy as np
import torchvision.transforms as trans
from torch.utils.data.dataset import Dataset
from skimage.color import hsv2rgb

from .utils import class_property


class ColoredSprites(Dataset):
    files = {"train": "data/raw/dsprites/colored_dsprites_train.npz"}

    n_factors = 6

    factors = ('color', 'shape', 'scale', 'orientation', 'posX', 'posY')

    factor_sizes = np.array([6, 3, 6, 20, 16,16])

    categorical = np.array([0, 1, 0, 0, 0, 0])

    img_size = (3, 64, 64)



    unique_values = {'posX': np.array([0.0, 0.06451613, 0.12903226, 0.19354839,
                                       0.25806452, 0.32258065, 0.38709677,
                                       0.4516129, 0.51612903, 0.58064516,
                                       0.64516129, 0.70967742, 0.77419355,
                                       0.83870968, 0.90322581,
                                       0.96774194]),
                  'posY': np.array([0., 0.06451613, 0.12903226, 0.19354839,
                                    0.25806452, 0.32258065, 0.38709677,
                                    0.4516129, 0.51612903, 0.58064516,
                                    0.64516129, 0.70967742, 0.77419355,
                                    0.83870968, 0.90322581,
                                    0.96774194]),
                  'scale': np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.]),
                  'orientation': np.array([0.0, 0.32221463, 0.64442926,
                                           0.96664389, 1.28885852, 1.61107316,
                                           1.93328779, 2.25550242, 2.57771705,
                                           2.89993168, 3.22214631, 3.54436094,
                                           3.86657557, 4.1887902, 4.51100484,
                                           4.83321947, 5.1554341, 5.47764873,
                                           5.79986336, 6.12207799]),
                  # square = 1, ellipsis = 2, heart = 3
                  'shape': np.array([1., 2., 3.]),
                  'color': np.linspace(0, 1, 10, endpoint=False)
              }

    def __init__(self,
        masks: np.ndarray,
        hues: np.ndarray,
        factor_values: np.ndarray,
        factor_classes: np.ndarray,
        _getter: Optional[Callable] = None,
    ):
        self.masks = masks
        self.hues = hues
        self.factor_values = factor_values
        self.factor_classes = factor_classes
        self._getter = _getter

        image_transforms = [
            trans.Lambda(hsv2rgb),
            trans.ToTensor(),
        ]

        self.transform = trans.Compose(image_transforms)

    def __getitem__(self, idx):
        img = np.zeros((64, 64, 3), dtype=np.float32)

        nonzero = self.masks[idx]
        hue = self.hues[idx] / 255

        img[nonzero[0], nonzero[1], 0] = hue
        img[nonzero[0], nonzero[1], 1:] = 1.0

        data = (
            self.transform(img),
            self.factor_values[idx],
            self.factor_classes[idx]
        )

        if self._getter is None:
            return data

        return self._getter(*data)

    def __len__(self):
        return len(self.masks)

    def __str__(self) -> str:
        return 'dSprites'


    @staticmethod
    def load_raw(path, factor_filter=None):
        data_zip = h5py.File(path, 'r')

        mask = [i.nonzero() for i in data_zip['mask'][()]]
        hue = data_zip['hue']
        factor_values = data_zip['latent_values']
        factor_classes = data_zip['latent_classes']

        if factor_filter is not None:
            idx = factor_filter(factor_values, factor_classes)

            mask = [mask[i] for i in list(idx.nonzero()[0])]
            hue = hue[idx]
            factor_values = factor_values[idx]
            factor_classes = factor_classes[idx]

            if len(mask) == 0:
                raise ValueError('Incorrect masking removed all data')

        return mask, hue, factor_values, factor_classes



class _splits:
    col, shp, scl, rot, tx, ty = 0, 1, 2, 3, 4, 5
    a90, a120, a180, a240 = np.pi / 2, 4 * np.pi / 3, np.pi, 2 * np.pi / 3

    @class_property
    def remove_redundant_rots(cls):
        def modifier(factor_values, factor_classes):
            square_rot = ((factor_values[:, cls.shp] == 1) &
                    (factor_values[:, cls.rot] < cls.a90))
            ellipsis_rot = ((factor_values[:, cls.shp] == 2) &
                    (factor_values[:, cls.rot] < cls.a180))
            all_hearts = factor_values[:, cls.shp] == 3

            return square_rot | ellipsis_rot | all_hearts

        return modifier

    # masks for blank right side condition
    @class_property
    def two_shapes_two_colors(cls):
        def test_mask(factor_values, factor_classes):
            ellipsis_and_blue = (
                (factor_values[:, cls.shp] == 1) &
                (factor_values[:, 0] < 0.5)
            )
            heart_and_red = (
                (factor_values[:, cls.shp] == 2) &
                (factor_values[:, 0] > 0.5)
            )
            return ellipsis_and_blue | heart_and_red

        def train_mask(factor_values, factor_classes):
            return ~train_mask(factor_values, factor_classes)

        return train_mask, test_mask

    @class_property
    def color_shape_only(cls):
        def center_only(factor_values, factor_classes):
            return (
                (factor_values[:, cls.rot] == 0) &
                (factor_classes[:, cls.tx] == 16) &
                (factor_classes[:, cls.ty] == 16)
            )

        def heart_and_color(factor_values):
            # print(np.unique(factor_values[:, cls.col]))
            # exit()
            return (
                (factor_values[:, cls.shp] == 3) &
                (factor_values[:, cls.col] > 0.5)
            )

        def test_mask(factor_values, factor_classes):
            return (
                heart_and_color(factor_values) &
                center_only(factor_values, factor_classes)
            )

        def train_mask(factor_values, factor_classes):
            return (
               ~heart_and_color(factor_values) &
               center_only(factor_values, factor_classes)
           )

        return train_mask, test_mask


splits = {
    "interp": {},

    "loo": {
    },

    "combgen": {
        'two_shapes_two_colors': _splits.two_shapes_two_colors,
        'shape_and_color_only': _splits.color_shape_only
    },

    "extrap": {
    },
}

modifiers = {
    'remove_redundant_rotations': _splits.remove_redundant_rots,
}
