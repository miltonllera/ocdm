import numpy as np
# import matplotlib.pyplot as plt


def strip(ax=None):
    if ax == None:
        ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

