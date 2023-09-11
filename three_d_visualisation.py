import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


class ThreeDVisualisation:

    def __init__(self, sample_dimensions, reconstruction_slices, step_size):
        """
        reconstruction slice data format must follow that of sample dimensions.
        if recon slices given in format xyz, sample dims must also be in xyz, and vice versa.
        step size remains constant throughout - artificially pad arrays if needed (temporary).

        :param sample_dimensions:
        :param reconstruction_slices:
        :param step_size:
        """
        self.dimensions = sample_dimensions
        self.data = reconstruction_slices
        self.step_size = step_size

        self._create_empty_grid()
        self._assign_values_to_grid()

    def _create_empty_grid(self):
        dims = self.dimensions
        step = self.step_size

        dims_shifted = [np.arange(-x/2, x/2 + x/(step * 4), step) for x in dims]
        x, y, z = np.meshgrid(dims_shifted[0], dims_shifted[1], dims_shifted[2])
        vertices = get_vertices(dims_shifted)

        vis_empty_grid = [np.linspace(min(x), max(x)) for x in dims_shifted]
        print(np.shape(vis_empty_grid))
        vis_empty_grid_ones = np.ones(np.shape(vis_empty_grid)[1])

        self.empty_grid = dims_shifted

    def _assign_values_to_grid(self):
        grid = self.empty_grid
        data = self.data

        for i, data_row in enumerate(data):
            for j, val in enumerate(data_row[0]):
                print(val)
                print(grid[i][j])

    def _visual(self):
        slices = self.data
        step_size = self.step_size

        ax = plt.figure().add_subplot(projection='3d')

        for recon in data:
            pass


def get_vertices(coords):
    maxima = np.asarray([[max(x), min(x)] for x in coords]).reshape((3, 2))
    print(maxima)

    # print(maxima)
    # vertices = np.zeros((8, 3))
    # half_len = int((np.shape(vertices)[0] / 2))
    #
    # vertices[:half_len] = maxima[0][0]
    # vertices[half_len:] = maxima[0][1]
    #
    # remaining_vertices = maxima[1:]

    rows = np.shape(maxima)[0]
    cols = np.shape(maxima)[1]
    vertices = []
    for i in range(rows):
        vertex = []
        for j in range(cols):
            idx = (i + rows) % rows
            jdx = (j + cols) % cols
            vertex.append(maxima[idx, jdx])
            # print(maxima[idx, jdx])
            # print(maxima[i, j])
            # print(maxima[i + 1, j])
            # print(maxima[i + 2, j])
        vertices.append(vertex)
    print(vertices)
    breakpoint()
            # vertices.append(maxima)

    breakpoint()


if __name__ == '__main__':
    dims = [60, 30, 20]
    step = 5
    data_points = [int(x/step) for x in dims]
    data = [np.random.randint(0, 100, data_point) for data_point in data_points]
    ThreeDVisualisation(dims, data, step)
