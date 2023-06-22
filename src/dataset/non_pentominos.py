import numpy as np
from src.dataset.utils import class_property
from .pentominos import Pentominos


class NonPentominos(Pentominos):
    """
    Use the same class for the non-pentominos and just override the methods used to load the
    different conditions.
    """
    @staticmethod
    def get_splits():
        return {
            'combgen': { },
            'extrap': { }
        }

    @staticmethod
    def get_modifiers():
        return { }


class _masks:
    shp, hue, scl, rot, tx, ty = 0, 1, 2, 3, 4, 5
    # def remove_redundant_rotations(cls):
    #     def modifier(factor_values, factor_classes):
    #         i_rotations = (
    #             (factor_values[:, cls.shp] == 0) &
    #             (factor_values[:, cls.rot] < 180)
    #         )

    #         x_rotations = (
    #             (factor_values[:, cls.shp] == 9) &
    #             (factor_values[:, cls.rot] < 90)
    #         )

    #         z_rotations = (
    #             (factor_values[:, cls.shp] == 11) &
    #             (factor_values[:, cls.rot] < 180)
    #         )

    #         rest = ~np.isin(factor_values[:, cls.shp], [0, 9, 11])

    #         return i_rotations | z_rotations | x_rotations | rest

    #     return modifier

    @class_property
    def exclude_donut(cls):
        """
        donut = 2, ellipses = 6
        """
        def test_mask(factor_values, factor_classes):
            excluded_shapes = factor_values[:, cls.shp] == 2
            return excluded_shapes

        def train_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return train_mask, test_mask

    @class_property
    def exclude_beignet(cls):
        def test_mask(factor_values, factor_classes):
            excluded_shapes = factor_values[:, cls.shp] == 3
            return excluded_shapes

        def train_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return train_mask, test_mask

    @class_property
    def exclude_heart(cls):
        def test_mask(factor_values, factor_classes):
            excluded_shapes = factor_values[:, cls.shp] == 1
            return excluded_shapes

        def train_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return train_mask, test_mask

    @class_property
    def exclude_star(cls):
        def test_mask(factor_values, factor_classes):
            excluded_shapes = factor_values[:, cls.shp] == 0
            return excluded_shapes

        def train_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return train_mask, test_mask

    @class_property
    def exclude_hexagon(cls):
        def test_mask(factor_values, factor_classes):
            excluded_shapes = factor_values[:, cls.shp] == 7
            return excluded_shapes

        def train_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return train_mask, test_mask

    @class_property
    def exclude_star_and_hexagon(cls):
        def test_mask(factor_values, factor_classes):
            excluded_shapes = np.isin(factor_values, [0, 7])
            return excluded_shapes
        def train_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)
        return train_mask, test_mask
