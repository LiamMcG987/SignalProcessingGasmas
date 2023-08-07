import os
import sys
import numpy as np
from numpy import matrix
from scipy import sparse
from scipy.sparse import linalg
from numpy.random import rand
import matplotlib.pyplot as plt

from utils import read_input_file

# PyToast environment

exec(compile(open(os.getenv("TOASTDIR") + "/ptoast_install.py", "rb").read(), os.getenv("TOASTDIR") + "/ptoast_install.py", 'exec'))
import toast

# Set the file paths
meshdir = os.path.expandvars("$TOASTDIR/test/2D/meshes/")
meshfile1 = meshdir + "ellips_tri10.msh"  # mesh for target data generation
meshfile2 = meshdir + "circle25_32.msh"   # mesh for reconstruction
qmfile = meshdir + "circle25_32x32.qm"    # source-detector file
muafile = meshdir + "tgt_mua_ellips_tri10.nim" # nodal target absorption
musfile = meshdir + "tgt_mus_ellips_tri10.nim" # nodal target scattering


class TomographicReconstruction:

    def __init__(self, data_files):
        self.gasmas_files = data_files

    def _read_images(self):
        gasmas_data = read_input_file(self.gasmas_files)
        self.gasmas_data = gasmas_data
        print(gasmas_data)

    def _mesh(self):
        meshdir = os.path.expandvars("$TOASTDIR/test/2D/meshes/")
        meshfile = meshdir + "circle25_32.msh"
        qmfile = meshdir + "circle25_32x32.qm"
        # hmesh = mesh.Read(meshfile)


if __name__ == '__main__':
    test = 'input_gasmas/measurements/230523/DAS_oxy_60_data_2023-05-23_144708.txt'
    TomographicReconstruction(test)
