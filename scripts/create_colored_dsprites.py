from argparse import ArgumentParser

import h5py
import numpy as np
from skimage.color import hsv2rgb


BATCH = 10_000


def main(args):
    with h5py.File(args.src, 'r') as f:
        imgs = np.asarray(f['imgs'])                              # (N, 64, 64), values in {0, 1}
        factor_values = np.asarray(f['latents']['values'])[:, 1:]  # (N, 5), float64
        factor_classes = np.asarray(f['latents']['classes'])[:, 1:]  # (N, 5), int64

    strides = [args.stride_shape, args.stride_scale, args.stride_orientation,
               args.stride_posX, args.stride_posY]
    keep = np.ones(len(imgs), dtype=bool)
    for fi, stride in enumerate(strides):
        if stride > 1:
            keep &= (factor_classes[:, fi] % stride == 0)
    imgs = imgs[keep]
    factor_values = factor_values[keep]
    factor_classes = factor_classes[keep]

    masks = imgs.astype(bool)
    n = len(masks)
    palette = np.linspace(0, 1, args.n_colors, endpoint=False)
    total = n * args.n_colors

    with h5py.File(args.dst, 'w') as out:
        ds_imgs = out.create_dataset(
            'imgs', shape=(total, 64, 64, 3), dtype=np.uint8, compression='gzip'
        )
        ds_vals = out.create_dataset('latent_values', shape=(total, 6), dtype=np.float64)
        ds_cls = out.create_dataset('latent_classes', shape=(total, 6), dtype=np.int64)

        for ci, hue in enumerate(palette):
            print(f"Color {ci + 1}/{args.n_colors} (hue={hue:.3f})")
            for start in range(0, n, BATCH):
                end = min(start + BATCH, n)
                b = end - start

                rgb = np.zeros((b, 64, 64, 3), dtype=np.float32)
                rgb[masks[start:end]] = [hue, 1.0, 1.0]
                rgb_uint8 = (hsv2rgb(rgb) * 255).astype(np.uint8)

                vals = np.concatenate(
                    [factor_values[start:end], np.full((b, 1), hue)], axis=1
                )
                cls = np.concatenate(
                    [factor_classes[start:end], np.full((b, 1), ci, dtype=np.int64)], axis=1
                )

                out_start = ci * n + start
                out_end = out_start + b
                ds_imgs[out_start:out_end] = rgb_uint8
                ds_vals[out_start:out_end] = vals
                ds_cls[out_start:out_end] = cls


parser = ArgumentParser()
parser.add_argument('--src', type=str, required=True, help='path to source dsprites HDF5')
parser.add_argument('--dst', type=str, required=True, help='path for output HDF5')
parser.add_argument('--n_colors', type=int, default=10, help='number of hue values')
parser.add_argument('--stride_shape', type=int, default=1)
parser.add_argument('--stride_scale', type=int, default=1)
parser.add_argument('--stride_orientation', type=int, default=1)
parser.add_argument('--stride_posX', type=int, default=1)
parser.add_argument('--stride_posY', type=int, default=1)


if __name__ == '__main__':
    main(parser.parse_args())
