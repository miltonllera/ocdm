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
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import h5py
import torchvision.transforms as trans
from skimage.color import rgb2hsv, rgb2gray
from skimage.feature import canny
from torch.utils.data import Dataset


def add_edges(x):
    edges = canny(rgb2gray(x), sigma=3)
    return np.concatenate([x, edges], axis=-1)


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
    factors = ('floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation')
    factor_sizes = np.array([10, 10, 10, 8, 4, 15])
    categorical = np.array([0, 0, 0, 0, 1, 0])
    img_size = (3, 64, 64)
    shape_names = np.asarray(["square", "cylinder", "sphere", "pill"])

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

    def __init__(
        self,
        path: str | None = None,
        batch_type: Literal[
            "unsupervised",
            "regress_latent_values",
            "regress_latent_classes"
        ] = "unsupervised",
        data_filter: Callable | None = None,
        color_format: Literal["rgb", "hsv"] = "rgb",
    ):
        if path is None:
            path = Path(self.files['train'])

        self.batch_type = batch_type
        self.images, self.factor_values, self.factor_classes = self.load_raw(path, data_filter)

        # image_transforms = [trans.ToTensor(), trans.Resize((124, 124)), trans.RandomCrop(64)]
        image_transforms = [trans.Lambda(add_edges), trans.ToTensor()]
        if color_format == 'hsv':
            image_transforms = [trans.Lambda(rgb2hsv)] + image_transforms
        self.transform = trans.Compose(image_transforms)

    def __getitem__(self, idx):
        img = self.transform(self.images[idx])
        if self.batch_type == "unsupervised":
            return img, img
        elif self.batch_type == "regression":
            return img, self.factor_values[idx]
        elif self.batch_type == "classification":
            return img, self.factor_classes[idx].astype(np.int64)

    def __len__(self):
        return len(self.images)

    def map_shapes(self, values):
        return type(self).shape_names[values.astype(int)]

    def __str__(self) -> str:
        return '3DShapes'

    @staticmethod
    def load_raw(path, factor_filter=None):
        with h5py.File(path, 'r') as data_zip:
            imgs = np.asarray(data_zip['images'])
            factor_values = np.asarray(data_zip['labels'])  # type: ignore

        factor_classes = np.asarray(list(product(
            *[range(i) for i in Shapes3D.factor_sizes])))

        if factor_filter is not None:
            idx = factor_filter(factor_values)

            imgs = imgs[idx]
            factor_values = factor_values[idx]
            factor_classes = factor_classes[idx]

            if len(imgs) == 0:
                raise ValueError('Incorrect masking removed all data')

        return imgs, factor_values, factor_classes
