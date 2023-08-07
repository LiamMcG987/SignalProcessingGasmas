import numpy as np
import matplotlib as mpl
from matplotlib import pyplot

from data_check import show_data


def create_blank_2d_grid(data):
    dims = len(data)
    zeros_grid = np.zeros(shape=(dims, dims))

    for i in range(dims):
        print(data)
        print(data * 1 + (i / 20))
        zeros_grid[i, :] = data * (1 + (i / 20))
        # zeros_grid[i, :] = data

    # mid_point = dims // 2
    # zeros_grid[:, mid_point] = data
    show_data(zeros_grid)
    return zeros_grid


def grid_creation(data):
    return create_blank_2d_grid(data)


if __name__ == '__main__':
    o2_data = [1, 2, 5, 2, 1]
    grid_creation(o2_data)
