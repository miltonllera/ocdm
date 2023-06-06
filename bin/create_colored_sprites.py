from argparse import ArgumentParser

import h5py
import numpy as np
# import matplotlib.pyplot as plt
# from skimage.color.colorconv import hsv2rgb

import src.dataset.dsprites as dsprites
from src.dataset.ood_loader import compose


filters = compose(dsprites.modifiers['halve_rots'], dsprites.modifiers['halve_pos'])


def generate_images(
    images: np.ndarray,
    palette: np.ndarray,
    path: str,
) -> None:
    # hsv[:, 1:] = 1.0
    # rgb = hsv2rgb(hsv)
    n_colors = len(palette)

    hsv = (palette * 255).astype(np.uint8).repeat(len(images), axis=0)
    mask = images[np.newaxis].repeat(
        n_colors, axis=0).reshape(-1, 64, 64).astype(bool)

    # fig, axes = plt.subplots(1, 10)

    # masks = [i.nonzero() for i in images]
    # for mask, hue, ax in zip(masks[::100000], hsv[::100000], axes):
    #     img = np.zeros((64, 64, 3), dtype=np.float32)

    #     img[mask[0], mask[1], 0] = hue / 255
    #     img[mask[0], mask[1], 1:] = 1.0
    #     ax.imshow(hsv2rgb(img))

    # fig.savefig("/home/milton/Dropbox/temp.png")

    with h5py.File(path, 'w') as data_zip:
        data_zip.create_dataset('mask', data=mask, compression="gzip")
        data_zip.create_dataset('hue', data=hsv)


def generate_latents(
    factor_values: np.ndarray,
    factor_classes: np.ndarray,
    palette: np.ndarray,
    path: str,
) -> None:
    n_colors = len(palette)
    factor_values = np.concatenate(
        [
            palette.repeat(len(factor_values))[:, np.newaxis],
            factor_values.repeat(n_colors, axis=0)
        ], axis=1
    )

    color_classes = np.arange(n_colors, dtype=factor_classes.dtype)
    factor_classes = np.concatenate(
        [
            color_classes.repeat(len(factor_classes))[:, np.newaxis],
            factor_classes.repeat(n_colors, axis=0)
        ], axis=1
    )

    with h5py.File(path, 'a') as data_zip:
        data_zip.create_dataset('latent_values', data=factor_values)
        data_zip.create_dataset('latent_classes', data=factor_classes)


def main(args):
    new_path = args.path.replace("dsprites_train", "colored_dsprites_train")
    images, factor_values, factor_classes = dsprites.DSprites.load_raw(
        args.path, filters)

    palette = np.linspace(0, 1, args.n_colors, endpoint=False)

    generate_images(images, palette, new_path)
    generate_latents(factor_values, factor_classes, palette, new_path)


parser = ArgumentParser()

parser.add_argument("--path", type=str, help="path to dsprites.")
parser.add_argument("--n_colors", type=int, default=10, help="number of colores used.")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
