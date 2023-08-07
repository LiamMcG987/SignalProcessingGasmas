import os.path

import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot


def get_data(file):
    file_no_ext = os.path.splitext(file)[0] + '256'
    data = scipy.io.loadmat(file)[file_no_ext]
    return data


def show_data(data):
    cmap = mpl.colors.ListedColormap(['white', 'green', 'yellow'])

    max_row_global, min_row_global = -np.Inf, np.Inf
    for row in data:
        max_row, min_row = max(row), min(row)
        if max_row > max_row_global:
            max_row_global = max_row
        if min_row < min_row_global:
            min_row_global = min_row

    img = pyplot.imshow(data, interpolation='nearest',
                        cmap=cmap)
    plt.show()


def main(file):
    recon_data = get_data(file)
    show_data(recon_data)
    # print(recon_data)


if __name__ == '__main__':
    test_file = 'phantom.mat'
    main(test_file)
