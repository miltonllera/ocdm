import os
import argparse

import pandas as pd

import matplotlib.pyplot as plt
from PIL.Image import open as img_open

from .extra.visualization import strip_plot


def plot_size(s):
    try:
        h, w = map(int, s.split(','))
        return h, w
    except:
        raise argparse.ArgumentTypeError("Plot size must be h, w")


parser = argparse.ArgumentParser(description='Summarize results for a condition')


parser.add_argument('--model_folders', type=str, nargs='+', dest='folders',
    help='Folders with saved model results')
parser.add_argument('--titles', type=str, nargs='+', dest='titles', default=[],
    help='Names to assign to each folder\'s subplots')
parser.add_argument('--to_aggregate', nargs="+", type=str,
    help="plots to aggregate into a comparison plot")
parser.add_argument("--plot_sizes", nargs="+", type=plot_size,
    help="Size of the grid for each subplot")
parser.add_argument('--save', type=str, required=True,
    help='Folder to save the plot')
parser.add_argument('--name', type=str, required=True, help='Plot name')


read_img = lambda folder, file: img_open(os.path.join(folder, "analysis", file))
read_csv = lambda folder, file: pd.read_csv(
    os.path.join(folder, "analysis", file), index_col=0
)


def main(folders, titles, to_aggregate, plot_sizes, save_folder, figure_name):
    plot_h, plot_w = zip(*plot_sizes)

    fig = plt.figure(layout="constrained", figsize=(sum(plot_w), sum(plot_h)))
    grd = plt.GridSpec(sum(plot_h), sum(plot_w), figure=fig)

    row = 0
    for plot_name, ps in zip(to_aggregate, plot_sizes):
        imgs_per_folder = [get_file_name(f, plot_name) for f in folders]

        row_height, col_width = ps

        column = 0
        for i, img in enumerate(imgs_per_folder):
            ax = fig.add_subplot(
                grd[row: row + row_height, column: column + col_width]
            )

            ax.imshow(img)

            strip_plot(ax)

            column += col_width

            if row == 0 and len(titles):
                ax.set_title(titles[i], fontsize=8)

        row += row_height

    os.makedirs(save_folder, exist_ok=True)
    full_path = os.path.join(save_folder, figure_name)
    fig.savefig(full_path, bbox_inches='tight', dpi=300)


def get_file_name(folder, plot_name):
    return read_img(folder, f"{plot_name}.png")

if __name__ == "__main__":
    args = parser.parse_args()

    main(
        args.folders,
        args.titles,
        args.to_aggregate,
        args.plot_sizes,
        args.save,
        args.name
    )
