"""
Disentanglement dataset from in Gondal et al 2019.

This dataset contains more realistic stimuli when compared to dSprites and
3Dshapes. Plus some combinations of factors have a tighther coupling between
them than others, which means that the models have an harder/easier time
learning how they interact.

For more info, the dataset can be found here:

arXiv preprint https://arxiv.org/abs/1906.03292
NeurIPS Challenge: https://www.aicrowd.com/challenges/
                           neurips-2019-disentanglement-challenge
"""

from typing import Callable, Literal, Optional

import numpy as np
import torch
import torchvision.transforms as trans
from itertools import product
from skimage.color import rgb2hsv
from torch.utils.data.dataset import Dataset
from .utils import class_property


class MPI3D(Dataset):
    """
    #==========================================================================
    # factor Dimension,    factor values                                 N vals
    #==========================================================================

    # object color:        white=0, green=1, red=2, blue=3,                  6
    #                      brown=4, olive=5
    # object shape:        cone=0, cube=1, cylinder=2,                       6
    #                      hexagonal=3, pyramid=4, sphere=5
    # object size:         small=0, large=1                                  2
    # camera height:       top=0, center=1, bottom=2                         3
    # background color:    purple=0, sea green=1, salmon=2                   3
    # horizontal axis:     40 values liearly spaced [0, 39]                 40
    # vertical axis:       40 values liearly spaced [0, 39]                 40
    """
    files = {"toy": "data/raw/mpi/mpi3d_toy.npz",
             "realistic": "data/raw/mpi/mpi3d_realistic.npz",
             "real": "data/raw/mpi/mpi3d_real.npz"}

    n_factors = 7
    factors = ('object_color', 'object_shape', 'object_size', 'camera_height',
               'background_color', 'horizontal_axis', 'vertical_axis')
    factor_sizes = np.array([6, 6, 2, 3, 3, 40, 40])

    categorical = np.array([0, 1, 1, 1, 1, 0, 0])

    img_size = (3, 64, 64)

    unique_values = { 'object_color'    : np.arange(6),
                      'object_shape'    : np.arange(6),
                      'object_size'     : np.arange(2),
                      'camera_height'   : np.arange(3),
                      'background_color': np.arange(3),
                      'horizontal_axis' : np.arange(40),
                      'vertical_axis'   : np.arange(40)}

    def __init__(self,
        images: np.ndarray,
        factor_values: np.ndarray,
        factor_classes: np.ndarray,
        version: Literal["real", "simulated", "toy"],
        color_format: Literal["rgb", "hsv"] = "rgb",
        _getter: Optional[Callable] = None,
    ):
        self.images = images
        self.factor_values = factor_values
        self.factor_classes = factor_classes
        self.version = version

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
        return 'MPI3D<{}>'.format(self.version)

    @staticmethod
    def load_raw(path, factor_filter=None):
        data_zip = np.load(path, allow_pickle=True)
        images = data_zip['images']

        factor_values = list(product(*MPI3D.unique_values.values()))
        factor_values = np.asarray(factor_values, dtype=np.int8)
        factor_classes = np.asarray(list(product(
           *[range(i) for i in MPI3D.factor_sizes])))

        if factor_filter is not None:
            idx = factor_filter(factor_values,factor_classes)

            images = images[idx]
            factor_values = factor_values[idx]

            if len(images) == 0:
                raise ValueError('Incorrect masking removed all data')

        return images, factor_values.astype(np.float32), factor_values


class _splits:
    oc, shp, sz, camh, bkg, hx, vx = 0, 1, 2, 3, 4, 5, 6

    # Modifiers
    @class_property
    def remove_redundant_shapes(cls):
        def modifier(factor_values, factor_classes):
            return ~np.isin(factor_values[:, cls.shp], np.array([0,3]))
        return modifier

    @class_property
    def fix_hx(cls):
        def modifier(factor_values, factor_classes):
            return factor_values[:, cls.hx] == 0
        return modifier

    @class_property
    def lhalf_hx(cls):
        def modifier(factor_values, factor_classes):
            return factor_values[:, cls.hx] < 20
        return modifier

    @class_property
    def even_vx(cls):
        def modifier(factor_values, factor_classes):
            return factor_classes[:, cls.vx] % 2 == 1
        return modifier

    # Extrapolation
    @class_property
    def exclude_horz_gt20(cls):
        def train_mask(factor_values, factor_classes):
            return factor_values[:, cls.hx] < 20

        def test_mask(factor_values, factor_classes):
            return (factor_values[:, cls.hx] > 20)

        return train_mask, test_mask

    @class_property
    def exclude_objc_gt3(cls):
        def train_mask(factor_values, factor_classes):
            return factor_values[:, cls.oc] <= 3

        def test_mask(factor_values, factor_classes):
            return factor_values[:, cls.oc] > 3

        return train_mask, test_mask

    # Recombination to element
    @class_property
    def cylinder_to_horizontal_axis(cls):
        def test_mask(factor_values, factor_classes):
            return ((factor_values[:, cls.hx] < 20) &
                    (factor_values[:, cls.shp] == 2))

        def training_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return training_mask, test_mask

    @class_property
    def cylinder_to_vertial_axis(cls):
        def test_mask(factor_values, factor_classes):
            return ((factor_values[:, cls.vx] < 20) &
                    (factor_values[:, cls.shp] == 2))

        def training_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return training_mask, test_mask

    @class_property
    def cylinder_to_background(cls):
        def test_mask(factor_values, factor_classes):
            return ((factor_values[:, cls.bkg] > 0) &
                    (factor_values[:, cls.shp] == 2))

        def training_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return training_mask, test_mask

    @class_property
    def redobject2hz(cls):
        def test_mask(factor_values, factor_classes):
            return ((factor_values[:, cls.hx] < 20) &
                    (factor_values[:, cls.oc] == 2))

        def training_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return training_mask, test_mask

    @class_property
    def background_to_cylinder(cls):
        def test_mask(factor_values, factor_classes):
            return ((factor_values[:, cls.shp] >= 4) &
                    (factor_values[:, cls.bkg] == 2))

        def train_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return train_mask, test_mask

    @class_property
    def background2obj_color(cls):
        def test_mask(factor_values, factor_classes):
            return ((factor_values[:, cls.bkg] == 2) &
                    (factor_values[:, cls.oc] > 2))

        def training_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return training_mask, test_mask

    # Recombination to element
    @class_property
    def leave1out(cls):
        def test_mask(factor_values, factor_classes):
            return ((factor_values[:, cls.oc] == 5) &
                    (factor_values[:, cls.shp] == 2) &
                    (factor_values[:, cls.sz] == 1) &
                    (factor_values[:, cls.camh] == 1) &
                    (factor_values[:, cls.bkg] == 1) &
                    (factor_values[:, cls.hx] > 35) &
                    (factor_values[:, cls.vx] > 35))

        def train_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return train_mask, test_mask


splits = {
    'interp': {},

    'loo': {
        'leave1out'  : _splits.leave1out
    },

    'combgen': {
        'cyl2hx': _splits.cylinder_to_horizontal_axis,
        'cyl2vx': _splits.cylinder_to_vertial_axis,
        'redoc2hx': _splits.redobject2hz,
        'bkg2cyl': _splits.background_to_cylinder
    },

    'extrap': {
        'horz_gt20'  : _splits.exclude_horz_gt20,
        'objc_gt3'   : _splits.exclude_objc_gt3
    },
}

modifiers= {
    'four_shapes': _splits.remove_redundant_shapes,
    'fix_hx'     : _splits.fix_hx,
    'lhalf_hx'   : _splits.lhalf_hx,
    'even_vx'    : _splits.even_vx
}
