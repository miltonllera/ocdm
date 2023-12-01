import argparse
from typing import Union
from enum import Enum

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import measure
from matplotlib.path import Path
from src.dataset.sprite_generator import SpriteDict, SpriteGenerator


def main(args):
    sprite_loader = SpriteLoader()
    SpriteGenerator(
        60,
        60,
        (2, 2),
        args.bg,
        args.lim_angles,
        args.num_angles,
        args.lim_scales,
        args.num_scales,
        args.lim_colors,
        args.num_colors,
        args.lim_xs,
        args.num_xs,
        args.lim_ys,
        args.num_ys,
        10,
        canonical_interpolation=Image.LANCZOS
    )(sprite_loader, args.folder)


class SpriteLoader(SpriteDict):
    class Names(Enum):
        HEART = 0
        STAR = 1
        DONUT = 2
        BEIGNET = 3
        SNAKE = 4
        ELLIPSE = 5
        HEXAGON = 6
        B = 7

    def __len__(self):
        return 8

    def __getitem__(self, entry: Union[int, str]):
        if isinstance(entry, str):
            entry = SpriteLoader.Names[entry]
        return self.get_shape(entry)

    def get_shape(self, shape):
        if shape == 0:  # star
            img = plt.imread("data/assets/non_pentominos/star_large.png")[..., 0].T
        elif shape == 1:
            img = plt.imread("data/assets/non_pentominos/heart_large.png")[..., 0].T
        elif shape == 2:
            img = plt.imread("data/assets/non_pentominos/donut_large.png")[..., 0].T
        elif shape == 3:
            img = plt.imread("data/assets/non_pentominos/beignet_large.png")[..., 0].T
        elif shape == 4:
            img = plt.imread("data/assets/non_pentominos/snake_large.png")[..., 0].T
        elif shape == 5:
            img = plt.imread("data/assets/non_pentominos/square_large.png")[..., 0].T
        elif shape == 6:
            img = plt.imread("data/assets/non_pentominos/ellipsis_large.png")[..., 0].T
        elif shape == 7:
            img = plt.imread("data/assets/non_pentominos/hexagon_large.png")[..., 0].T
        elif shape == 8:
            img = plt.imread("data/assets/non_pentominos/b_large.png")[..., 0].T
        else:
            raise ValueError(f"Unrecognized shape code {shape}")

        img = np.pad(1 - img, pad_width=((2,2), (2,2)), mode='constant')
        contours = measure.find_contours(img, level=0.99)

        # _, ax = plt.subplots()
        # ax.imshow(img, cmap=plt.cm.gray)

        # for contour in contours:
        #     ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

        assert len(contours) == 1  # there should be only one shape

        vertices = contours[0]
        vertices = (vertices - vertices.min(axis=0)) * 10 / vertices.max()

        correction_mask = np.zeros_like(vertices)

        return Path(vertices), np.asarray(correction_mask, dtype=float)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("sprite generatror")

    parser.add_argument("--height", type=int, default=60,
        help="Height of the image")
    parser.add_argument("--width", type=int, default=None,
        help="Width of the image")
    parser.add_argument("--pad", type=int, default=(2, 2), nargs=2,
        help="Padding on each side of the image")
    parser.add_argument("--aa", type=int, default=10,
        help="Amount by which image is upscaled before performing transforms.")
    parser.add_argument("--bg", type=int, default=None, nargs=3,
        help="Background color. Defaults to black")
    parser.add_argument("--lim_angles", type=float, default=None, nargs=2,
        help="Rotation angle range")
    parser.add_argument("--num_angles", type=int, default=16,
        help="Number of rotation angles to sample")
    parser.add_argument("--lim_scales", type=float, default=None, nargs=2,
        help="Scale value range")
    parser.add_argument("--num_scales", type=int, default=5,
        help="Number of scale values to sample")
    parser.add_argument("--lim_colors", type=float, default=None, nargs=2,
        help="Hue range in HSV format")
    parser.add_argument("--num_colors", type=int, default=1,
        help="Number of scale values to sample")
    parser.add_argument("--lim_xs", type=float, default=None, nargs=2,
        help="Hue range in HSV format")
    parser.add_argument("--num_xs", type=int, default=10,
        help="Number of horizontal position values to sample")
    parser.add_argument("--lim_ys", type=float, default=None, nargs=2,
        help="Hue range in HSV format")
    parser.add_argument("--num_ys", type=int, default=10,
        help="Number of vertical position values to sample")
    parser.add_argument("--folder", type=str,
        help="Folder where to save the dataset")

    args = parser.parse_args()
    main(args)
