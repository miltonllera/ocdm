"""
dSprites dataset module

The module contains the code for loading the dSprites dataset. This dataset
contains transformations of simple sprites in 2 dimensions, which have no
detailed features.

The original dataset can be found at:
    https://github.com/deepmind/3d-shapes
"""


from typing import Optional, Callable
import numpy as np
import torch
import torchvision.transforms as trans
from torch.utils.data.dataset import Dataset

from .utils import class_property


class DSprites(Dataset):
    """
    Disentanglement dataset from Loic et al, (2017).

    #==========================================================================
    # factor Dimension,    factor values                                 N vals
    #==========================================================================

    # shape:               1=heart, 2=ellipsis, 3=square                      3
    # scale                uniform in range [0.5, 1.0]                        6
    # orientation          uniform in range [0, 2 * pi]                      40
    # position x           unifrom in range [0, 1]                           36
    # position y           unifrom in range [0, 1]                           36
    """
    urls = {"train": "https://github.com/deepmind/dsprites-dataset/blob/master/"
                     "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true"}
    files = {"train": "data/raw/dsprites/dsprite_train.npz"}

    n_factors = 5

    factors = ('shape', 'scale', 'orientation', 'posX', 'posY')

    factor_sizes = np.array([3, 6, 40, 32, 32])

    categorical = np.array([1, 0, 0, 0, 0])

    img_size = (1, 64, 64)

    unique_values = {'posX': np.array([0., 0.03225806, 0.06451613, 0.09677419,
                                       0.12903226, 0.16129032, 0.19354839,
                                       0.22580645, 0.25806452, 0.29032258,
                                       0.32258065, 0.35483871, 0.38709677,
                                       0.41935484, 0.4516129, 0.48387097,
                                       0.51612903, 0.5483871, 0.58064516,
                                       0.61290323, 0.64516129, 0.67741935,
                                       0.70967742, 0.74193548, 0.77419355,
                                       0.80645161, 0.83870968, 0.87096774,
                                       0.90322581, 0.93548387, 0.96774194, 1.]),
                  'posY': np.array([0., 0.03225806, 0.06451613, 0.09677419,
                                    0.12903226, 0.16129032, 0.19354839,
                                    0.22580645, 0.25806452, 0.29032258,
                                    0.32258065, 0.35483871, 0.38709677,
                                    0.41935484, 0.4516129, 0.48387097,
                                    0.51612903, 0.5483871, 0.58064516,
                                    0.61290323, 0.64516129, 0.67741935,
                                    0.70967742, 0.74193548, 0.77419355,
                                    0.70967742, 0.74193548, 0.77419355,
                                    0.80645161, 0.83870968, 0.87096774,
                                    0.90322581, 0.93548387, 0.96774194, 1.]),
                  'scale': np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.]),
                  'orientation': np.array([0., 0.16110732, 0.32221463,
                                           0.48332195, 0.64442926, 0.80553658,
                                           0.96664389, 1.12775121, 1.28885852,
                                           1.44996584, 1.61107316, 1.77218047,
                                           1.93328779, 2.0943951, 2.25550242,
                                           2.41660973, 2.57771705, 2.73882436,
                                           2.89993168, 3.061039, 3.22214631,
                                           3.38325363, 3.54436094, 3.70546826,
                                           3.86657557, 4.02768289, 4.1887902,
                                           4.34989752, 4.51100484, 4.67211215,
                                           4.83321947, 4.99432678, 5.1554341,
                                           5.31654141, 5.47764873, 5.63875604,
                                           5.79986336, 5.96097068, 6.12207799,
                                           6.28318531]),
                  # square = 1, ellipsis = 2, heart = 3
                  'shape': np.array([1., 2., 3.]),
              }

    def __init__(self,
        images: np.ndarray,
        factor_values: np.ndarray,
        factor_classes: np.ndarray,
        _getter: Optional[Callable] = None,
    ):
        self.images = images
        self.factor_values = factor_values
        self.factor_classes = factor_classes
        self._getter = _getter

        image_transforms = [
            trans.ToTensor(),
            trans.ConvertImageDtype(torch.float32)
        ]

        self.transform = trans.Compose(image_transforms)

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
        return 'dSprites'

    @staticmethod
    def load_raw(path, factor_filter=None):
        data_zip = np.load(path, allow_pickle=True)

        imgs = data_zip['imgs'] * 255
        factor_values = data_zip['latents_values'][:, 1:]  # Remove luminescence
        factor_classes = data_zip['latents_classes'][:, 1:]

        if factor_filter is not None:
            idx = factor_filter(factor_values, factor_classes)

            imgs = imgs[idx]
            factor_values = factor_values[idx]
            factor_classes = factor_classes[idx]

            if len(imgs) == 0:
                raise ValueError('Incorrect masking removed all data')

        return imgs, factor_values, factor_classes


class _splits:
    shp, scl, rot, tx, ty = 0, 1, 2, 3, 4
    a90, a120, a180, a240 = np.pi / 2, 4 * np.pi / 3, np.pi, 2 * np.pi / 3

    @class_property
    def sparse_posX(cls):
        def modifier(factor_values, factor_classes):
            return np.isin(
                factor_classes[:, cls.tx],
                np.asarray([0, 7, 15, 23, 31])
            )
        return modifier

    @class_property
    def sparse_posY(cls):
        def modifier(factor_values, factor_classes):
            return np.isin(
                factor_classes[:, cls.ty],
                np.asarray([0, 7, 15, 23, 31])
            )
        return modifier

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
    def blank_side(cls):
        def blank_side_train(factor_values, factor_classes):
            return (factor_values[:, cls.tx] < 0.5)

        def blank_side_extrp(factor_values, factor_classes):
            return (factor_values[:, cls.tx] > 0.5)

        return blank_side_train, blank_side_extrp

    # Leave one shape out along translation dimension
    @class_property
    def square_to_posX(cls):
        def shape2tx_train(factor_values, factor_classes):
            return ((factor_values[:, cls.shp] != 1) |
                    (factor_values[:, cls.tx] < 0.5))

        def shape2tx_extrp(factor_values, factor_classes):
            return ((factor_values[:, cls.shp] == 1) &
                    (factor_values[:, cls.tx] > 0.5))

        return shape2tx_train, shape2tx_extrp

    @class_property
    def ellipse_to_posX(cls):
        def shape2tx_train(factor_values, factor_classes):
            return ((factor_values[:, cls.shp] != 2) |
                    (factor_values[:, cls.tx] < 0.5))

        def shape2tx_extrp(factor_values, factor_classes):
            return ((factor_values[:, cls.shp] == 2) &
                    (factor_values[:, cls.tx] > 0.5))

        return shape2tx_train, shape2tx_extrp

    # leave1out_comb
    @class_property
    def leave1out(cls):
        def leave1out_comb_test(factor_values, factor_classes):
            return ((factor_classes[:, cls.shp] == 1) &
                    (factor_values[:, cls.scl] > 0.6) &
                    (factor_values[:, cls.rot] > 0.0) &
                    (factor_values[:, cls.rot] < cls.a120) &
                    (factor_values[:, cls.rot] > cls.a240) &
                    (factor_values[:, cls.tx] > 0.66) &
                    (factor_values[:, cls.ty] > 0.66))

        def leave1out_comb_train(factor_values, factor_classes):
            return ~leave1out_comb_test(factor_values, factor_classes)

        return leave1out_comb_train, leave1out_comb_test

    @class_property
    def square_to_scale(cls):
        def shape2tx_train(factor_values, factor_classes):
            return ((factor_values[:, cls.shp] != 1) |
                    (factor_values[:, cls.scl] < 0.75))

        def shape2tx_extrp(factor_values, factor_classes):
            return ((factor_values[:, cls.shp] == 1) &
                    (factor_values[:, cls.scl] > 0.75))

        return shape2tx_train, shape2tx_extrp

    @class_property
    def centered_sqr2tx(cls):
        def test_mask(factor_values, factor_classes):
            return ((factor_values[:, cls.shp] == 1) &
                    (factor_values[:, cls.tx] > 0.25) &
                    (factor_values[:, cls.tx] < 0.75))

        def train_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return train_mask, test_mask

    @class_property
    def rshift_sqr2tx(cls):
        def test_mask(factor_values, factor_classes):
            return ((factor_values[:, cls.shp] == 1) &
                    (factor_values[:, cls.tx] > 0.40) &
                    (factor_values[:, cls.tx] < 0.90))

        def train_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return train_mask, test_mask

    @class_property
    def flanked_sqr2tx(cls):
        def train_mask(factor_values, factor_classes):
            return ((factor_values[:, cls.shp] != 1) |
                    (factor_values[:, cls.tx] < 0.25) |
                    (factor_values[:, cls.tx] > 0.75))

        def test_mask(factor_values, factor_classes):
            return ~train_mask(factor_values, factor_classes)

        return train_mask, test_mask

    @class_property
    def lshift_sqrt2tx(cls):
        def train_mask(factor_values, factor_classes):
            return ((factor_values[:, cls.shp] != 2) |
                    ((factor_values[:, cls.tx] > 0.10) &
                     (factor_values[:, cls.tx] < 0.60)))

        def test_mask(factor_values, factor_classes):
            return ~train_mask(factor_values, factor_classes)

        return train_mask, test_mask

    @class_property
    def heart_to_rot(cls):
        def train_mask(factor_values, factor_classes):
            return ((factor_values[:, cls.shp] != 3) |
                    (factor_values[:, cls.rot] < cls.a180))

        def test_mask(factor_values, factor_classes):
            return ~train_mask(factor_values, factor_classes)

        return train_mask, test_mask


splits = {
    "interp": {},

    "loo": {
        'leave1out'  : _splits.leave1out
    },

    "combgen": {
        'sqr2tx'     : _splits.square_to_posX,
        'ell2tx'     : _splits.ellipse_to_posX,
        'sqr2scl'    : _splits.square_to_scale,
        'sqr2tx_cent': _splits.centered_sqr2tx,
        'sqr2tx_flnk': _splits.flanked_sqr2tx,
        'sqr2tx_rs'  : _splits.rshift_sqr2tx,
        'sqr2tx_ls'  : _splits.lshift_sqrt2tx,
        'heart2rot'  : _splits.heart_to_rot,
    },

    "extrap": {
        'blank_side' : _splits.blank_side
    },
}

modifiers = {
    'sparse_posX': _splits.sparse_posX,
    'remove_redundant_rotations': _splits.remove_redundant_rots
}
