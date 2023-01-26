import os
import argparse

import pandas as pd

import matplotlib.pyplot as plt
from PIL.Image import open as img_open

parser = argparse.ArgumentParser(description='Summarize results for a condition')


parser.add_argument('--model_folders', type=str, nargs='+', dest='folders',
                    help='Folders with saved model results')
parser.add_argument('--to_aggregate', nargs='+', type=str,
                    help="plots to aggregate into a comparison plot")
parser.add_argument('--save', type=str, required=True,
                    help='Folder to save the plot')
parser.add_argument('--name', type=str, required=True, help='Plot name')


read_img = lambda folder, file: img_open(os.path.join(folder, file))
read_csv = lambda folder, file: pd.read_csv(os.path.join(folder, file),
                                            index_col=0)


def strip_plot(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.set_xlabel('')
    ax.set_ylabel('')

    ax.set_xticks([])
    ax.set_yticks([])


def main(folders, to_aggregate, save, name):
    model_ids = [os.path.split(p)[1] for p in folders]

    fig = plt.figure(constrained_layout=True, figsize=(15, 5))
    gs = plt.GridSpec(2 * len(to_aggregate), len(model_ids) * 6, figure=fig)

    for i, plot_name in enumerate(to_aggregate):
        plots = [read_img(f, plot_name) for f in folders]
        for j, img in enumerate(plots):
            ax = fig.add_subplot(
                gs[2 * i: 2 * i + 1, j * 6: (j + 1) * 6]
            )

            ax.imshow(img)
            strip_plot(ax)

    os.makedirs(save, exist_ok=True)
    fig.savefig(os.path.join(save, name), bbox_inches='tight', dpi=300)


if __name__ == "__main__":
    args = parser.parse_args()

    folders = args.folders
    save = args.save
    name = args.name
    to_aggregate = args.to_aggregate

    main(folders, to_aggregate, save, name)
