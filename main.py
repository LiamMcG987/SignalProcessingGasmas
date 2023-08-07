import os
import math
import scipy
import numpy as np

from reference_calculations.air_reference import AirReference
from gasmas_calculations.gasmas_signal import GasmasSignalProcessing
from gasmas_calculations.oxygen_concentration_calibrated import calc_gasmas_signal
from astra_sample.astra_sample import sinogram_2d
from reconstruction_calculations.direct_fourier_recon import DFR2D
from utils import read_input_file, read_conditions, read_data_from_folders

# from three_d_visualisation import ThreeDVisualisation

if __name__ == '__main__':
    ref_h2o_files = 'input_gasmas/measurements/water_vapour/'
    ref_o2_files = 'input_gasmas/measurements/oxygen/'
    conditions = 'input_gasmas/measurements/conditions/conditions_1205.xlsx'

    h20_params = [2.98E9, 1.936E-11]
    o2_params = [1.59E9, 2.36E-13]

    rh_atm, t_atm = read_conditions(conditions)
    # rh_atm, t_atm = 69.26, 20.5
    pressure = 1020.25
    # pressure = 1006.3
    rh_sample = 95

    recon_iterations = 20

    calibrations = AirReference(ref_h2o_files, ref_o2_files, t_atm, rh_atm, pressure)
    calibration_constant = calibrations.calibration_constant_avg

    # data_folder = '../../Measurements/test_folder_360'
    data_folder = '../../Measurements/1907'
    # data_folder = '../../Measurements/cassava'
    gasmas_o2_files, gasmas_h2o_files, tomo_angles = read_data_from_folders(data_folder, angle_interval=15)
    o2_concs_grid = []
    projections_all = np.zeros(len(gasmas_o2_files))

    for j in range(len(gasmas_h2o_files)):
        current_h2o_file = gasmas_h2o_files[j]
        current_o2_file = gasmas_o2_files[j]

        gasmas_data_h20, _time, _pathlength = read_input_file(current_h2o_file)
        gasmas_data_o2, __time, __pathlength = read_input_file(current_o2_file)
        o2_concs, projections_local = np.zeros(len(gasmas_data_h20)), []

        for i in range(len(gasmas_data_h20)):
            print(os.listdir(current_h2o_file)[i])
            print(os.listdir(current_o2_file)[i])
            try:
                h20 = gasmas_data_h20[i]
                o2 = gasmas_data_o2[i]
                gasmas_h20 = GasmasSignalProcessing(h20, plots=True)
                gasmas_o2 = GasmasSignalProcessing(o2, plots=False)

                transmittance_peak_h20, transmittance_peak_o2 = gasmas_h20.transmittance_peak, \
                    gasmas_o2.transmittance_peak
                peaks = [transmittance_peak_h20, transmittance_peak_o2]

                o2_concs[i] = calc_gasmas_signal(h20_params, o2_params, peaks, calibration_constant, t_atm,
                                                 pressure, rh_sample)

                projection = np.asarray(gasmas_o2.absorbance_profile)
                projections_local.append(projection)
            except TypeError:
                print('Unable to extract data from certain dataset. Arbitrary value assigned for projection.')
                projections_local.append(np.zeros(256))
                continue
            except IndexError:
                print('Different number of datasets provided.')
                continue

        # print(o2_concs)
        # breakpoint()
        DFR2D(projections_local, tomo_angles, recon_iterations)
        # breakpoint()
        o2_concs_grid.append(o2_concs)

        o2_concs_grid_np = np.asarray(o2_concs).reshape(-1, 1)
        # sinogram_2d(o2_concs_grid_np, tomo_angles)

    print(o2_concs_grid)
    height = 10
