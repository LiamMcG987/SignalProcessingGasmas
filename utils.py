import os
import re
import shutil
import math
import numpy as np
import pandas as pd
from scipy import constants
from pathlib import Path


def read_input_file(file, wms=False):
    if os.path.isdir(file):
        return read_directory(file)

    with open(file) as fin:
        lines = fin.readlines()

    measurements_list = []
    for line in lines:
        if 'timestamp' in line.lower() or line.startswith('\n'):
            continue

        line_split = str(line).split(';')
        if wms:
            measurements = line_split[-3]
        else:
            measurements = line_split[-2]
        measurements_list.append([eval(x) for x in measurements.strip().split(' ')])

    input_signal = np.asarray(measurements_list[0])
    signal_time = np.arange(1, len(input_signal) + 1)
    print(f'data extracted from {os.path.split(file)[-1]}')

    physical_pathlength = re.search(r'\d+', os.path.split(file)[-1])
    if physical_pathlength is None:
        physical_pathlength = 1
    else:
        physical_pathlength = int(physical_pathlength.group())

    return [input_signal], [signal_time], physical_pathlength


def read_conditions(file):
    conds = pd.read_excel(file, index_col=0)
    conds_df = pd.DataFrame(conds.T)

    rh_df, temp_df = conds_df['RH'], conds_df['temp']
    rh, temp = np.average([x for x in rh_df]), np.average([(x + 273) for x in temp_df])
    return rh, temp


def read_directory(diry):
    cwd = os.getcwd()
    diry_path = os.path.join(cwd, diry)
    dir_files = sorted(os.listdir(diry_path))

    diry_data = np.zeros(shape=(len(dir_files), 256))
    phy_pathlengths = np.zeros(shape=len(dir_files))

    for i, file in enumerate(dir_files):
        file_path = os.path.join(diry_path, file)

        signal, time, phy_pathlength = read_input_file(file_path)
        phy_pathlengths[i] = phy_pathlength
        diry_data[i] = signal[0]

    return diry_data, np.arange(1, len(diry_data[0]) + 1), phy_pathlengths


def beer_lambert(output: str, gamma_v, s, trans_min, t_atm, pressure, pathlength=None, c=None):
    boltzmann = constants.Boltzmann * 1E6

    n_0 = 101325 / (boltzmann * t_atm)
    # n_0 = pressure * 1000 / (boltzmann * t_atm)
    g = 1 / (np.pi * gamma_v)

    if output.lower() == 'concentration':
        exponent = -s * g * pathlength
        n = np.log(trans_min) / exponent
        conc = n / n_0
        return conc
    elif output.lower() == 'pathlength':
        n = c * n_0
        exponent = -s * g * n
        pathlength = np.log(trans_min) / exponent
        return pathlength
    else:
        raise ValueError('Incorrect output specified. Must be pathlength or concentration.')


def clean_folder(folder):
    new_folders = ['oxygen', 'water_vapour', 'params']

    if any(not os.path.isdir(x) for x in new_folders):
        for new_folder in new_folders:
            Path(f'{folder}/{new_folder}').mkdir(exist_ok=True)

    oxy_path = f'{folder}/oxygen'
    water_path = f'{folder}/water_vapour'
    params_path = f'{folder}/params'

    for file in os.listdir(folder):
        file_path = f'{folder}/{file}'

        if 'params' in str(file).lower():
            shutil.move(file_path, params_path)
        elif 'oxy' in str(file).lower():
            shutil.move(file_path, oxy_path)
        else:
            shutil.move(file_path, water_path)


def read_data_from_folders(master_folder, angle_interval=None, fan_beam=False):
    oxygen_data_list, vapour_data_list = [], []

    for folder in os.listdir(master_folder):
        clean_folder(f'{master_folder}/{folder}')

        oxygen_data_list.append(f'{master_folder}/{folder}/oxygen/')
        vapour_data_list.append(f'{master_folder}/{folder}/water_vapour/')

    folder_len = len(oxygen_data_list[0])
    if angle_interval is not None:
        end_angle = int(angle_interval) * folder_len
        angles = np.linspace(0, end_angle, folder_len, endpoint=True)
    else:
        if not fan_beam:
            angles = np.asarray([int(re.search(r'\d+', x).group()) for x in os.listdir(oxygen_data_list[0])])
        else:
            angles = np.unique([int(re.search(r'\d+', x).group()) for x in os.listdir(oxygen_data_list[0])])
            return [math.radians(x) for x in sorted(angles)]

    angles_rads = [math.radians(x) for x in sorted(angles)]
    return sorted(oxygen_data_list), sorted(vapour_data_list), angles_rads


if __name__ == '__main__':
    folder_test = '../../Measurements/test_folder'
    angle_increment = None
    oxy, vap, angles_tomo = read_data_from_folders(folder_test, angle_increment)
