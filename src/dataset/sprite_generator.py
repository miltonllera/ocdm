from abc import ABC, abstractmethod
import os
import os.path as osp
import json
import multiprocessing as mp
from itertools import product
from typing import Optional, Tuple

import numpy as np
import colorsys
from PIL import Image, ImageDraw
from matplotlib.path import Path
from matplotlib.transforms import Affine2D


class SpriteDict(ABC):
    @abstractmethod
    def get_shape(self, int):
        raise NotImplementedError()

    @abstractmethod
    def __len__(self):
        raise NotImplementedError()


class SpriteGenerator:
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
        chaikin_iters: int = 5,
        # rendering options
        anti_aliasing: int = 10,
        interpolation = Image.LANCZOS,
        canonical_interpolation = Image.NEAREST,
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
        self.chaikin_iters = chaikin_iters

        # rendering options
        self.aa = anti_aliasing
        self.interpolation = interpolation
        self.canonical_interp = canonical_interpolation
        self._save_folder = None
        self._sprite_dict = None

    @property
    def image_size(self) -> Tuple[int, int]:
        w = self.non_padded_size[0] + 2 * self.padding[0]
        h = self.non_padded_size[1] + 2 * self.padding[1]
        return h, w

    def __call__(self, sprite_dict: SpriteDict, folder: str) -> None:
        image_folder = osp.join(folder, "images")
        os.makedirs(image_folder, exist_ok=True)

        self._save_folder = image_folder
        self._sprite_dict = sprite_dict

        shapes = np.array(list(range(len(sprite_dict))))
        colors = np.linspace(*self.lim_colors, self.num_colors, endpoint=True)
        scales = np.linspace(*self.lim_scales, self.num_scales, endpoint=True)
        angles = np.linspace(*self.lim_angles, self.num_angles, endpoint=True)
        pos_xs = np.linspace(*self.lim_xs, self.num_xs, endpoint=True)
        pos_ys = np.linspace(*self.lim_ys, self.num_ys, endpoint=True)

        # shapes = np.array([4.])
        # colors = np.array([0.])
        # scales = np.array([1.5])
        # # angles = np.array([0.])
        # angles = np.linspace(0, 360, 16, endpoint=True)
        # pos_xs = np.array([0.])
        # pos_ys = np.array([0.])

        all_combinations = product(
            shapes, colors, scales, angles, pos_xs, pos_ys
        )

        N_CPUS = mp.cpu_count() - 1

        if N_CPUS > 1:
            n_combinations = (
                len(shapes) * len(colors) * len(scales) * len(angles) * len(pos_xs) * len(pos_ys)
            )

            with mp.Pool(N_CPUS) as pool:
                value_dict = pool.imap(
                    self.generate_instance,
                    enumerate(all_combinations),
                    int(np.ceil(n_combinations / N_CPUS)),
                )

                value_dict = list(value_dict)

        else:
            value_dict = []
            for params in enumerate(all_combinations):
                value_dict.append(self.generate_instance(params))

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

    def generate_image(self, shape_dict, shape, color, scale, angle, tx, ty, value=1.0):
        canvas_w = self.aa * self.non_padded_size[0]
        canvas_h = self.aa * self.non_padded_size[1]

        # Create polygon.
        # Because of how PIL works, we must keep track of which vertices define
        # sides that face right and/or down. These need to be moved back by one
        # pixel, regardless of image size, to preserve the length of the sides.
        # See: https://github.com/python-pillow/Pillow/issues/7104
        sprite, pixel_correction_mask = shape_dict[shape]

        # Apply scaling independently and correct for the extra pixel AFTERWARDS!.
        center_and_scale = Affine2D().translate(-5, -5).scale(self.aa * scale)

        sprite = center_and_scale.transform_path(sprite)
        sprite = Path(sprite.vertices - pixel_correction_mask)

        if self.chaikin_iters > 0:
            sprite = Path(chaikins_corner_cutting(sprite.vertices, self.chaikin_iters))

        scaled_tx = self.aa * tx + canvas_w / 2
        scaled_ty = self.aa * ty + canvas_h / 2
        rotate_and_translate = (
            Affine2D().rotate_deg(angle).translate(scaled_tx, scaled_ty)
        )

        vertices = rotate_and_translate.transform_path(sprite).vertices

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
            interp = self.canonical_interp
        else:
            interp = self.interpolation

        canvas = canvas.resize(self.non_padded_size, resample=interp)

        # pad image
        image = Image.new('RGB', self.image_size, color=self.bg_color)
        image.paste(canvas, self.padding)

        return image

    def generate_instance(self, params):
        i, (shp, col, scl, ang, tx, ty) = params

        image_file = f"image_{i:08}.png"

        image = self.generate_image(self._sprite_dict, shp, col, scl, ang, tx, ty)
        image.save(osp.join(self._save_folder, image_file))

        return {
            'image_file_name': image_file,
            'shape': int(shp),
            'color': col,
            'scale': scl,
            'angle': ang,
            'pos_x': tx,
            'pos_y': ty
        }


def hsv_to_rgb(c):
  """Convert HSV tuple to RGB tuple."""
  return tuple((255 * np.array(colorsys.hsv_to_rgb(*c))).astype(np.uint8))


def chaikins_corner_cutting(coords, refinements=5):
    coords = np.array(coords)

    for _ in range(refinements):
        L = coords.repeat(2, axis=0)
        R = np.empty_like(L)
        R[0] = L[0]
        R[2::2] = L[1:-1:2]
        R[1:-1:2] = L[2::2]
        R[-1] = L[-1]
        coords = L * 0.75 + R * 0.25

    return coords
