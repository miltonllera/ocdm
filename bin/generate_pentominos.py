"""
Generate sprite-like dataset using Pentomino shapes.

The advantage of this dataset is that it has more shapes than similar ones such
as dSprites (12 vs 3) and the shapes share features that only vary in their
spatial configuration (they are all composed of squares).

Current implementation ignores mirrored versions of some shapes for simplicity.

More details can be found at:

    https://en.wikipedia.org/wiki/Pentomino

"""

import argparse
from typing import Union
from enum import Enum

import numpy as np
from matplotlib.path import Path
from src.dataset.sprite_generator import SpriteDict, SpriteGenerator


def main(args):
    SpriteGenerator(
        args.height,
        args.width,
        args.pad,
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
        0,  # number of corner cutting iterations.
        args.aa,
    )(PentominoLoader(), args.folder)


class PentominoLoader(SpriteDict):
    class Names(Enum):
        I = 0
        F = 1
        L = 2
        P = 3
        N = 4
        T = 5
        U = 6
        V = 7
        W = 8
        X = 9
        Y = 10
        Z = 11

    def __len__(self):
        return 12

    def __getitem__(self, entry: Union[int, str]):
        if isinstance(entry, str):
            entry = PentominoLoader.Names[entry]
        return self.get_shape(entry)

    def get_shape(self, shape):
        """
        Pentomino shapes as a matplotlib polygon. Names in Conway's convention.
        """
        if shape == 0: # I
            """
            00100
            00100
            00100
            00100
            00100
            """
            vertices = [(4, 0), (6, 0), (6, 10), (4, 10)]
            correction_mask = [(0, 0), (1, 0), (1, 1), (0, 1)]

        elif shape == 1: # F
            """
            00000
            00110
            01100
            00100
            00000
            """
            vertices = [
                (4, 2), (8, 2), (8, 4), (6, 4), (6, 8),
                (4, 8), (4, 6), (2, 6), (2, 4), (4, 4)
            ]

            correction_mask = [
                (0, 0), (1, 0), (1, 1), (1, 1), (1, 1),
                (0, 1), (0, 1), (0, 1), (0, 0), (0, 0)
            ]

        elif shape == 2: # L
            """
            00000
            00100
            00100
            00100
            00110
            """
            vertices = [(3, 1), (5, 1), (5, 7), (7, 7), (7, 9), (3, 9)]
            correction_mask = [(0, 1), (1, 0), (1, 0), (1, 0), (1, 1), (0, 1)]

        elif shape == 3: # P
            """
            00000
            01100
            01100
            01000
            00000
            """
            vertices = [(3, 2), (7, 2), (7, 6), (5, 6), (5, 8), (3, 8)]
            correction_mask = [(0, 0), (1, 0), (1, 1), (1, 1), (1, 1), (0, 1)]

        elif shape == 4: # N
            """
            00000
            00110
            11100
            00000
            00000
            """
            vertices = [
                (5, 3), (9, 3), (9, 5), (7, 5), (7, 7), (1, 7), (1, 5), (5, 5)
            ]
            correction_mask = [
                (0, 1), (1, 0), (1, 1), (1, 1), (0, 1), (0, 1), (0, 0), (0, 0)
            ]

        elif shape == 5: # T
            """
            00000
            01110
            00100
            00100
            00000
            """
            vertices = [
                (2, 2), (8, 2), (8, 4), (6, 4), (6, 8), (4, 8), (4, 4), (2, 4)
            ]
            correction_mask =[
                (0, 0), (1, 0), (1, 1), (1, 1), (1, 1), (0, 1), (0, 1), (0, 1)
            ]

        elif shape == 6: # U
            """
            00000
            01010
            01110
            00000
            00000
            """
            vertices = [
                (2, 3), (4, 3), (4, 5), (6, 5), (6, 3), (8, 3), (8, 7), (2, 7)
            ]
            correction_mask = [
                (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (1, 0), (1, 1), (0, 1)
            ]

        elif shape == 7: # V
            """
            00000
            00010
            00010
            01110
            00000
            """
            vertices = [(2, 6), (6, 6), (6, 2), (8, 2), (8, 8), (2, 8)]
            correction_mask = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0), (0, 0)]

        elif shape == 8: # W
            """
            00000
            00010
            00110
            01100
            00000
            """
            vertices = [
                (2, 6), (4, 6), (4, 4), (6, 4), (6, 2),
                (8, 2), (8, 6), (6, 6), (6, 8), (2, 8)
            ]
            correction_mask = [
                (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
                (1, 0), (1, 1), (1, 1), (1, 1), (0, 1)
            ]

        elif shape == 9: # X
            """
            00000
            00100
            01110
            00100
            00000
            """
            vertices = [
                (2, 4), (4, 4), (4, 2), (6, 2), (6, 4), (8, 4),
                (8, 6), (6, 6), (6, 8), (4, 8), (4, 6), (2, 6)
            ]
            correction_mask =[
                (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (1, 0),
                (1, 1), (1, 1), (1, 1), (0, 1), (0, 1), (0, 1)
            ]

        elif shape == 10: # Y
            """
            00000
            00100
            01100
            00100
            00100
            """
            vertices = [
                (3, 3), (5, 3), (5, 1), (7, 1), (7, 9), (5, 9), (5, 5), (3, 5)
            ]
            correction_mask = [
                (0, 0), (0, 0), (0, 0), (1, 0), (1, 1), (0, 1), (0, 1), (0, 1)
            ]

        elif shape == 11: # Z
            """
            00000
            01100
            00100
            00110
            00-00
            """
            vertices = [
                (2, 2), (6, 2), (6, 6), (8, 6), (8, 8), (4, 8), (4, 4), (2, 4)
            ]
            correction_mask = [
                (0, 0), (1, 0), (1, 0), (1, 0), (1, 1), (0, 1), (0, 1), (0, 1)
            ]

        # chiral (mirrowed) pentominos

        # elif shape == 12: # R (mirrored F)
        #     """
        #     00000
        #     01100
        #     00110
        #     00100
        #     00000
        #     """

        # elif shape == 13: # J (mirrored L)
        #     """
        #     00000
        #     00100
        #     00100
        #     00100
        #     01100
        #     """

        # elif shape == 14: # Q (mirrored P)
        #     """
        #     00000
        #     01100
        #     01100
        #     00100
        #     00000
        #     """

        # elif shape == 15: # mirrored N
        #     """
        #     00000
        #     01100
        #     00111
        #     00000
        #     00000
        #     """

        # elif shape == 16: # mirrored Y
        #     """
        #     00000
        #     00100
        #     00110
        #     00100
        #     00100
        #     """

        # elif shape == 17: # S (mirrored Z)
        #     """
        #     00000
        #     00110
        #     00100
        #     01100
        #     00-00
        #     """

        else:
            raise ValueError(f"Unrecognized shape code {shape}")

        return Path(vertices), np.asarray(correction_mask, dtype=float)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("pentomino generatror")

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
