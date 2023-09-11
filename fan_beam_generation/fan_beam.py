import os
import re
import numpy as np
import matplotlib.pyplot as plt

from utils import read_data_from_folders, read_input_file
from gasmas_calculations.gasmas_signal import GasmasSignalProcessing
from gasmas_calculations.oxygen_concentration_calibrated import calc_gasmas_signal


class FanBeam:

    def __init__(self, projections_folder, sample_diameter, angle_interval):
        self.projection_folder = projections_folder
        self.sample_diameter = sample_diameter
        self.angle_interval = angle_interval

        self.oxygen_projections, self.water_projections, self.angles_rads\
            = read_data_from_folders(self.projection_folder)
        self.projections = [self.oxygen_projections, self.water_projections]
        self._group_fan_projections()

    def _group_fan_projections(self):
        projection_folder = self.projection_folder
        master_dict = dict.fromkeys(os.listdir(projection_folder))
        for nested_dict in master_dict:
            master_dict[nested_dict] = dict.fromkeys(['oxygen', 'water_vapour'])

        for projection_group in self.projections:
            for folder in projection_group:
                peaks_global = []
                folder_split = os.path.normpath(folder).split(os.sep)
                top_level_folder = folder_split[-2]
                bottom_level_folder = folder_split[-1]

                angles = np.asarray([get_projection_rotation(x) for x in os.listdir(folder)])
                angles_unique = np.unique(angles[:, 0])

                grouped_files = group_files_w_angles(folder, angles_unique)
                for items in grouped_files.values():
                    peaks = []

                    for item in items:
                        gasmas_signal, _, _ = read_input_file(item)
                        peaks.append(GasmasSignalProcessing(gasmas_signal[0], plots=False).transmittance_peak)
                    # peak_positions = linear_superposition(peaks, self.angle_interval, self.sample_diameter, item)
                    peaks_global.append(peaks)
                master_dict.get(top_level_folder)[bottom_level_folder] = peaks_global
        self.peaks_dict = master_dict


def group_files_w_angles(folder, angles):
    grouped_files = dict.fromkeys(angles)
    for item in grouped_files:
        grouped_files[item] = []

    for dirpath, _, filenames in os.walk(folder):
        for f in filenames:
            matched_angle = get_angle_filename_match(f, angles)
            grouped_files[matched_angle].append(os.path.abspath(os.path.join(dirpath, f)))

    return grouped_files


def get_projection_rotation(filename):
    digits = [int(s) for s in filename.split('_') if s.isdigit()]
    return digits


def get_angle_filename_match(file, angles_unique):
    angle = int(re.findall(r'\d+', file)[0])
    if angle in angles_unique:
        return angle


def linear_superposition(peaks, angle_interval, sample_diameter, folder):
    folder_label = os.path.splitext(os.path.split(folder)[-1])[0][:-7]
    angle_rads = [np.deg2rad(x) for x in angle_interval]
    arc_length = [sample_diameter / 2 * x for x in angle_rads]
    peaks_positions = np.asarray([[i, j] for i, j in zip(arc_length, peaks)])
    plt.plot(peaks_positions[:, 0], peaks_positions[:, 1], '-.')
    plt.scatter(peaks_positions[:, 0], peaks_positions[:, 1], 20)
    plt.title(f'Linear Superposition: {folder_label}')
    plt.show()
    return peaks_positions


if __name__ == '__main__':
    proj = '../../../Measurements/2508_superimpose'
    interval = [0, 30, 60]
    diameter = 80
    fb = FanBeam(proj, diameter, interval)
