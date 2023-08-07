import matplotlib.pyplot as plt
import numpy as np


class ThreeDVisualisation:

    def __init__(self, data, height_interval):
        self.data = data
        self.height_interval = height_interval

        self._visual()

    def _visual(self):
        two_d_recons = self.data
        z_gap = self.height_interval

        ax = plt.figure().add_subplot(projection='3d')

        for recon in two_d_recons:
