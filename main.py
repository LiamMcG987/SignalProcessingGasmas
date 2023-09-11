import os

import matplotlib.pyplot as plt
import numpy as np

from reference_calculations.air_reference import AirReference
from gasmas_calculations.gasmas_signal import GasmasSignalProcessing
from gasmas_calculations.oxygen_concentration_calibrated import calc_gasmas_signal
from fan_beam_generation.fan_beam import FanBeam
from reconstruction_calculations.direct_fourier_recon import DFR2D
from utils import read_input_file, read_conditions, read_data_from_folders


def main(ref_h2o_files, ref_o2_files, conditions, data_folder, diameter,
         h2o_params=None, o2_params=None, pressure=None, rh_sample=None, iteration_number=None,
         fan_beam=False, fan_beam_folder=None, angle_interval=None):
    h2o_params = h2o_params if h2o_params is not None else [2.98E9, 1.936E-11]
    o2_params = o2_params if o2_params is not None else [1.59E9, 2.36E-13]
    pressure = pressure if pressure is not None else 1020.25
    rh_sample = rh_sample if rh_sample is not None else 95
    iteration_number = iteration_number if iteration_number is not None else 20

    rh_atm, t_atm = read_conditions(conditions)
    ref_properties = [t_atm, rh_atm, pressure]
    calibration_constant, poly_order_h2o, poly_order_o2 = calibration(ref_h2o_files, ref_o2_files, ref_properties)

    process_properties = [t_atm, pressure, rh_sample]
    if fan_beam:
        tomo_angles = read_data_from_folders(fan_beam_folder, fan_beam=True)
        gasmas_dict = FanBeam(fan_beam_folder, diameter, angle_interval).peaks_dict
        recons = process_peaks(gasmas_dict, [h2o_params, o2_params], calibration_constant, process_properties,
                               tomo_angles)
    else:
        gasmas_o2_files, gasmas_h2o_files, tomo_angles = read_data_from_folders(data_folder)
        recons = process_datasets(gasmas_h2o_files, gasmas_o2_files, h2o_params, o2_params, calibration_constant,
                                  process_properties,
                                  tomo_angles, iteration_number)

    visualise_reconstructions(recons)


def process_peaks(peaks_dict, params, constant, props, tomo_angles):
    h2o_params, o2_params = params
    t_atm, pressure, rh_sample = props

    recons = []
    for heading in peaks_dict.values():
        oxy = heading.get('oxygen')
        vap = heading.get('water_vapour')
        projections = np.zeros(np.shape(oxy))

        for i, fan_oxy in enumerate(oxy):
            fan_vap = vap[i]
            o2_concs, h2o_pls = np.zeros(len(fan_oxy)), np.zeros(len(fan_oxy))
            for j, peak_oxy in enumerate(fan_oxy):
                peak_vap = fan_vap[j]
                peaks = [peak_vap, peak_oxy]
                _, o2_concs[j] = calc_gasmas_signal(h2o_params, o2_params, peaks, constant,
                                                    t_atm, pressure, rh_sample)
            projections[i] = o2_concs
        recons.append(DFR2D(projections, tomo_angles, 20, constant).recon)
    return recons


def process_datasets(gasmas_h2o_files, gasmas_o2_files, h2o_params, o2_params, calibration_constant, properties,
                     tomo_angles, recon_iterations):
    t_atm, pressure, rh_sample = properties
    recons, o2_concs_grid = [], []

    for j in range(len(gasmas_h2o_files)):
        current_h2o_file = gasmas_h2o_files[j]
        current_o2_file = gasmas_o2_files[j]

        gasmas_data_h20, _time, _pathlength = read_input_file(current_h2o_file)
        gasmas_data_o2, __time, __pathlength = read_input_file(current_o2_file)
        o2_concs, projections_local, h20_pathlengths = np.zeros(len(gasmas_data_h20)), [], []

        for i in range(len(gasmas_data_h20)):
            print(os.listdir(current_h2o_file)[i])
            print(os.listdir(current_o2_file)[i])
            try:
                h2o = gasmas_data_h20[i]
                o2 = gasmas_data_o2[i]
                gasmas_h2o = GasmasSignalProcessing(h2o, plots=False)
                gasmas_o2 = GasmasSignalProcessing(o2, plots=False)

                transmittance_peak_h20, transmittance_peak_o2 = gasmas_h2o.transmittance_peak, \
                    gasmas_o2.transmittance_peak
                peaks = [transmittance_peak_h20, transmittance_peak_o2]

                h20_pathlength, o2_concs[i] = calc_gasmas_signal(h2o_params, o2_params, peaks, calibration_constant,
                                                                 t_atm, pressure, rh_sample)

                projection = np.asarray(gasmas_o2.absorbance_profile)
                projections_local.append(projection)
                h20_pathlengths.append(h20_pathlength)
            except TypeError:
                print('Unable to extract data from certain dataset. Arbitrary value assigned for projection.')
                projections_local.append(np.zeros(256))
                h20_pathlengths.append(np.Inf)
                continue
            except IndexError:
                print('Different number of datasets provided.')
                continue

        recons.append(DFR2D(projections_local, tomo_angles, recon_iterations, calibration_constant, h20_pathlengths)
                      .recon)
        o2_concs_grid.append(o2_concs)
        o2_concs_grid_np = np.asarray(o2_concs).reshape(-1, 1)

    return recons


def calibration(h2o, o2, properties):
    t_atm, rh_atm, pressure = properties

    calibrations = AirReference(h2o, o2, t_atm, rh_atm, pressure)
    calibration_constant = calibrations.calibration_constant_avg
    poly_order_h2o, poly_order_o2 = calibrations.opt_poly_order_h2o, calibrations.opt_poly_order_o2

    return calibration_constant, poly_order_h2o, poly_order_o2


def visualise_reconstructions(reconstructions):
    fig, ax = plt.subplots(len(reconstructions))
    for i, recon in enumerate(reconstructions):
        ax[i].imshow(np.real(recon))
        ax[i].set_title(f'Reconstruction #{i}')

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    reference_h2o_files = 'input_gasmas/measurements/water_vapour/'
    reference_o2_files = 'input_gasmas/measurements/oxygen/'
    reference_conditions = 'input_gasmas/measurements/conditions/conditions_1205.xlsx'
    gasmas_data_folder = '../../Measurements/1008'

    fb_folder = '../../Measurements/2508_superimpose'
    interval = [0, 30, 60]
    sample_diameter = 80

    main(reference_h2o_files, reference_o2_files, reference_conditions, gasmas_data_folder, sample_diameter,
         fan_beam=True, fan_beam_folder=fb_folder, angle_interval=interval)
