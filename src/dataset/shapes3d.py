"""
3DShapes dataset module

The module contains the code for loading the 3DShapes dataset. The dataset can
be loaded in 3 modes: supervised, unsupervised, and ground-truth factor
reconstruction. We mostly use the last for 3 training and the first one for
analyzing the results. Data loading of the batches is handled in the
corresponding Sacred ingredient.

The original dataset can be found at:
    https://github.com/deepmind/3d-shapes
"""
from itertools import product
from typing import Optional, Callable, Literal

import numpy as np
import h5py
import torch
import torchvision.transforms as trans
from skimage.color import rgb2hsv
from torch.utils.data import Dataset
from .utils import class_property
# from functools import partialmethod


class Shapes3D(Dataset):
    """
    Disentangled dataset used in Kim and Mnih, (2019)

    #==========================================================================
    # factor Dimension,    factor values                                 N vals
    #==========================================================================

    # floor hue:           uniform in range [0.0, 1.0)                      10
    # wall hue:            uniform in range [0.0, 1.0)                      10
    # object hue:          uniform in range [0.0, 1.0)                      10
    # scale:               uniform in range [0.75, 1.25]                     8
    # shape:               0=square, 1=cylinder, 2=sphere, 3=pill            4
    # orientation          uniform in range [-30, 30]                       15
    """
    files = {"train": "data/raw/shapes3d/3dshapes.h5"}

    n_factors = 6

    factors = ('floor_hue', 'wall_hue', 'object_hue',
               'scale', 'shape', 'orientation')

    factor_sizes = np.array([10, 10, 10, 8, 4, 15])

    categorical = np.array([0, 0, 0, 0, 1, 0])

    img_size = (3, 64, 64)

    unique_values = {'floor_hue': np.array([0., 0.1, 0.2, 0.3, 0.4,
                                            0.5, 0.6, 0.7, 0.8, 0.9]),
                     'wall_hue': np.array([0., 0.1, 0.2, 0.3, 0.4,
                                           0.5, 0.6, 0.7, 0.8, 0.9]),
                     'object_hue': np.array([0., 0.1, 0.2, 0.3, 0.4,
                                             0.5, 0.6, 0.7, 0.8, 0.9]),
                     'scale': np.array([0.75, 0.82142857, 0.89285714,
                                        0.96428571, 1.03571429, 1.10714286,
                                        1.17857143, 1.25]),
                     'shape': np.array([0, 1, 2, 3]),
                     'orientation': np.array([-30., -25.71428571, -21.42857143,
                                     -17.14285714, -12.85714286, -8.57142857,
                                     -4.28571429, 0., 4.28571429, 8.57142857,
                                     12.85714286, 17.14285714, 21.42857143,
                                     25.71428571,  30.])}

    def __init__(self,
        images: np.ndarray,
        factor_values: np.ndarray,
        factor_classes: np.ndarray,
        color_format: Literal["rgb", "hsv"] = "rgb",
        _getter: Optional[Callable] = None,
    ):
        self.images = images
        self.factor_values = factor_values
        self.factor_classes = factor_classes

        image_transforms = [
            trans.ToTensor(),
            trans.ConvertImageDtype(torch.float32)
        ]

        if color_format == 'hsv':
            image_transforms.insert(0, trans.Lambda(rgb2hsv))

        self.transform = trans.Compose(image_transforms)
        self._getter = _getter

    def __getitem__(self, idx):
        data = (
            self.transform(self.images[idx]),
            self.factor_values[idx],
            self.factor_classes[idx]
        )

        if self._getter is None:
            return data

        return self._getter(*data)

    def __len__(self):
        return len(self.images)

    def __str__(self) -> str:
        return '3DShapes'

    @staticmethod
    def load_raw(path, factor_filter=None):
        data_zip = h5py.File(path, 'r')

        imgs = data_zip['images'][()]
        factor_values = data_zip['labels'][()]
        factor_classes = np.asarray(list(product(
            *[range(i) for i in Shapes3D.factor_sizes])))

        if factor_filter is not None:
            idx = factor_filter(factor_values, factor_classes)

            imgs = imgs[idx]
            factor_values = factor_values[idx]
            factor_classes = factor_classes[idx]

            if len(imgs) == 0:
                raise ValueError('Incorrect masking removed all data')

        return imgs, factor_values, factor_classes


class _splits:
    """
    Boolean masks used to partition the Shapes3D dataset
    for each generalisation condition
    """

    fh, wh, oh, scl, shp, orient = 0, 1, 2, 3, 4, 5

    # Modifies
    @class_property
    def exclude_odd_ohues(cls):
        def modifier(factor_values, factor_classes):
            return (factor_classes[:, cls.oh] % 2) == 0
        return modifier

    @class_property
    def exclude_half_ohues(cls):
        def modifier(factor_values, factor_classes):
            return factor_classes[:, cls.oh] < 5
        return modifier

    @class_property
    def exclude_odd_wnf_hues(cls):
        def modifier(factor_values, factor_classes):
            return (((factor_classes[:, cls.wh] % 2) == 0) &
                    ((factor_classes[:, cls.fh] % 2) == 0))
            return modifier

    # Interpolation variants
    @class_property
    def odd_ohue(cls):
        def train_mask(factor_values, factor_classes):
            return factor_classes[:, cls.oh] % 2 == 0

        def test_mask(factor_values, factor_classes):
            return factor_classes[:, cls.oh] % 2 == 1

        return train_mask, test_mask

    @class_property
    def odd_wnf_hue(cls):
        def train_mask(factor_values, factor_classes):
            return cls.exclude_odd_wnf_hues(factor_values, factor_classes)

        def test_mask(factor_values, factor_classes):
            return ~cls.exclude_odd_wnf_hues(factor_values, factor_classes)

        return train_mask, test_mask

    # Extrapolation variants
    @class_property
    def missing_fh_50(cls):
        def train_mask(factor_values, factor_classes):
            return factor_values[:, cls.fh] < 0.5

        def test_mask(factor_values, factor_classes):
            return ~train_mask(factor_values, factor_classes)

        return train_mask, test_mask

    # Recombination to range
    @class_property
    def ohue_to_whue(cls):
        def test_mask(factor_values, factor_classes):
            return ((factor_values[:, cls.oh] >= 0.75) &
                    (factor_values[:, cls.wh] <= 0.5))

        def train_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return train_mask, test_mask

    @class_property
    def fhue_to_whue(cls):
        def test_mask(factor_values, factor_classes):
            return ((factor_values[:, cls.fh] >= 0.75) &
                    (factor_values[:, cls.wh] <= 0.5))

        def train_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return train_mask, test_mask

    @class_property
    def shape_to_floor(cls):
        def test_mask(factor_values, factor_classes):
            return ((factor_values[:, cls.shp] == 3.0) &
                    (factor_values[:, cls.fh] >= 0.5))

        def train_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return train_mask, test_mask

    @class_property
    def shape_to_objh(cls):
        def test_mask(factor_values, factor_classes):
            return ((factor_values[:, cls.shp] == 3.0) &
                    (factor_values[:, cls.oh] >= 0.5))

        def train_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return train_mask, test_mask

    @class_property
    def flanked_shape2ohue(cls):
        def test_mask(factor_values, factor_classes):
            return ((factor_values[:, cls.shp] == 3.0) &
                    (factor_values[:, cls.oh] > 0.25) &
                    (factor_values[:, cls.oh] < 0.75))

        def train_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return train_mask, test_mask

    @class_property
    def shape_to_objh_quarter(cls):
        def test_mask(factor_values, factor_classes):
            return ((factor_values[:, cls.shp] == 3.0) &
                    (factor_values[:, cls.oh] <= 0.25))

        def train_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return train_mask, test_mask

    @class_property
    def shape_to_orientation(cls):
        def test_mask(factor_values, factor_classes):
            return ((factor_classes[:,cls.shp] == 3.0) &
                    (factor_values[:,cls.orient] >= 0))

        def train_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return train_mask, test_mask

    # Recombination to element
    @class_property
    def leave1out(cls):
        def test_mask(factor_values, factor_classes):
            return ((factor_values[:, cls.oh] >= 0.8) &
                    (factor_values[:, cls.wh] >= 0.8) &
                    (factor_values[:, cls.fh] >= 0.8) &
                    (factor_values[:, cls.scl] >= 1.1) &
                    (factor_values[:, cls.shp] == 1) &
                    (factor_values[:, cls.orient] > 20))

        def train_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return train_mask, test_mask

    @class_property
    def shape_ohue(cls):
        def test_mask(factor_values, factor_classes):
            return ((factor_classes[:, cls.shp] == 3.0) &
                    (factor_classes[:, cls.oh] == 2))

        def train_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return train_mask, test_mask


splits = {
    'interp': {
        'odd_ohue'    : _splits.odd_ohue,
        'odd_wnf_hue' : _splits.odd_wnf_hue,
    },

    'loo' : {
        'shape2ohue' : _splits.shape_ohue,
        'leave1out'  : _splits.leave1out,
    },

    'combgen': {
        'ohue2whue'      : _splits.ohue_to_whue,
        'fhue2whue'      : _splits.fhue_to_whue,
        'shape2ohue'     : _splits.shape_to_objh,
        'shape2ohueq'    : _splits.shape_to_objh_quarter,
        'shape2fhue'     : _splits.shape_to_floor,
        'shape2orient'   : _splits.shape_to_orientation,
        'shape2ohue_flnk': _splits.flanked_shape2ohue
    },

    'extrap': {
        'missing_fh' : _splits.missing_fh_50
    },
}

modifiers = {
    'even_ohues'   : _splits.exclude_odd_ohues,
    'half_ohues'   : _splits.exclude_half_ohues,
    'even_wnf_hues': _splits.exclude_odd_wnf_hues
}
