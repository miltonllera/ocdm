"""
Generate sprite-like dataset using Pentomino shapes.

The advantage of this dataset is that it has more shapes than similar ones such
as dSprites (12 vs 3) and the shapes share features that only vary in their
spatial configuration (they are all composed of squares).

Current implementation ignores mirrored versions of some shapes for simplicity.

More details can be found at:

    https://en.wikipedia.org/wiki/Pentomino

"""

import os
import os.path as osp
import argparse
import json
from itertools import product
from typing import Optional, Tuple

import numpy as np
import colorsys
from PIL import Image, ImageDraw
from matplotlib.path import Path
from matplotlib.transforms import Affine2D


class PentominoGenerator:
    def __init__(
        self,
        # image properties
        height: int = 60,
        width: Optional[int] = None,
        padding: Tuple[int, int] = (2, 2),
        bg_color: Optional[Tuple[int, int, int]] = None,
        # factors of variation
        lim_angles: Optional[Tuple[float, float]] = None,
        num_angles: int = 40,
        lim_scales: Optional[Tuple[float, float]] = None,
        num_scales: int = 5,
        lim_colors: Optional[Tuple[float, float]] = None,
        num_colors: int = 1,
        lim_xs: Optional[Tuple[float, float]] = None,
        num_xs: int = 20,
        lim_ys: Optional[Tuple[float, float]] = None,
        num_ys: int = 20,
        # rendering options
        anti_aliasing: int = 10,
        interpolation = Image.LANCZOS,
    ):
        if width is None:
            width = height
        if bg_color is None:
            bg_color = (0, 0, 0)
        if lim_angles is None:
            lim_angles = (0, 360 * (1 - 1 / num_angles))
        if lim_scales is None:
            lim_scales = (1.5, 3.0)
        if lim_colors is None:
            lim_colors = (0, 1 - 1 / num_colors)
        if lim_xs is None:
            lim_xs = (-10, 10)
        if lim_ys is None:
            lim_ys = (-10, 10)

        # image properties
        self.non_padded_size = width, height
        self.padding = padding
        self.bg_color = bg_color

        # factors of variation
        self.lim_angles = lim_angles
        self.num_angles = num_angles
        self.lim_scales = lim_scales
        self.num_scales = num_scales
        self.lim_colors = lim_colors
        self.num_colors = num_colors
        self.lim_xs = lim_xs
        self.num_xs = num_xs
        self.lim_ys = lim_ys
        self.num_ys = num_ys

        # rendering options
        self.aa = anti_aliasing
        self.interpolation: Image._Resample = interpolation

    @property
    def image_size(self) -> Tuple[int, int]:
        w = self.non_padded_size[0] + 2 * self.padding[0]
        h = self.non_padded_size[1] + 2 * self.padding[1]
        return h, w

    def __call__(self, folder) -> None:
        image_folder = osp.join(folder, "images")
        os.makedirs(image_folder, exist_ok=True)

        shapes = np.array(list(range(12)))
        colors = np.linspace(*self.lim_colors, self.num_colors, endpoint=True)
        scales = np.linspace(*self.lim_scales, self.num_scales, endpoint=True)
        angles = np.linspace(*self.lim_angles, self.num_angles, endpoint=True)
        pos_xs = np.linspace(*self.lim_xs, self.num_xs, endpoint=True)
        pos_ys = np.linspace(*self.lim_ys, self.num_ys, endpoint=True)

        all_combinations = product(
            shapes, colors, scales, angles, pos_xs, pos_ys
        )

        value_dict = []

        for i, (shp, col, scl, ang, tx, ty) in enumerate(all_combinations):
            image_file = f"image_{i:08}.png"

            image = self.generate_image(shp, col, scl, ang, tx, ty)
            image.save(osp.join(image_folder, image_file))

            value_dict.append({
                'image_file_name': image_file,
                'shape': int(shp),
                'color': col,
                'scale': scl,
                'angle': ang,
                'pos_x': tx,
                'pos_y': ty
            })

        json_output = {
            'data': value_dict,
            'meta': {
                'height': int(self.image_size[0]),
                'width': int(self.image_size[1]),
                'anti_alias': self.aa,
                'bg_color': self.bg_color,
                'n_samples': len(value_dict),
                'factors': [
                    "shape", "color", "scale", "angle", "pos_x", "pos_y"
                ],
                'unique_values': {
                    'shapes': shapes.tolist(),
                    'colors': colors.tolist(),
                    'scales': scales.tolist(),
                    'angles': angles.tolist(),
                    'pos_xs': pos_xs.tolist(),
                    'pos_ys': pos_ys.tolist(),
                },
            }
        }

        with open(folder + "pentominos.json", mode="w+") as f:
            json.dump(json_output, f)

    def generate_image(self, shape, color, scale, angle, tx, ty, value=1.0):
        canvas_w = self.aa * self.non_padded_size[0]
        canvas_h = self.aa * self.non_padded_size[1]

        # Create polygon.
        # Because of how PIL works, we must keep track of which vertices define
        # sides that face left or down. These need to be moved back by one
        # pixel, regardless of image size, to preserve the length of the sides.
        # See: https://github.com/python-pillow/Pillow/issues/7104
        pentomino, pixel_correction_mask = self.get_shape(shape)

        # Apply scaling independently and correct for the extra pixel.
        center_and_scale = Affine2D().translate(-5, -5).scale(self.aa * scale)

        pentomino = center_and_scale.transform_path(pentomino)
        pentomino = Path(pentomino.vertices - pixel_correction_mask)

        scaled_tx = self.aa * tx + canvas_w / 2
        scaled_ty = self.aa * ty + canvas_h / 2
        rotate_and_translate = (
            Affine2D().rotate_deg(angle).translate(scaled_tx, scaled_ty)
        )

        vertices = rotate_and_translate.transform_path(pentomino).vertices

        # render image
        canvas = Image.new('RGB', (canvas_w, canvas_h), self.bg_color)
        draw = ImageDraw.Draw(canvas)

        # Use white if only one color is used.
        if self.num_colors == 1:
            color = (255, 255, 255)
        else:
            color = hsv_to_rgb((color * 360, 1.0, value))

        draw.polygon([(x, y) for x, y in vertices], fill=color)

        # If rotation is canonical, use NEAREST to obtain a crip shape.
        if angle in [0.0, 90.0, 180.0, 270.0]:
            interp = Image.NEAREST
        else:
            interp = self.interpolation

        canvas = canvas.resize(self.non_padded_size, resample=interp)

        # pad image
        image = Image.new('RGB', self.image_size, color=self.bg_color)
        image.paste(canvas, self.padding)

        return image

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


def hsv_to_rgb(c):
  """Convert HSV tuple to RGB tuple."""
  return tuple((255 * np.array(colorsys.hsv_to_rgb(*c))).astype(np.uint8))


def main(args):
    pentomino_generator = PentominoGenerator(
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
        args.aa,
    )
    pentomino_generator(args.folder)


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
