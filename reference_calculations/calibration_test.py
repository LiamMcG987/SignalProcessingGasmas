import numpy as np
from math import exp
import scipy.constants as constants

from utils import read_input_file


class CalibrationTest:

    def __init__(self, h2o, o2, conditions, cuts):
        self.h2o = read_input_files(h2o)[0][0]
        self.o2 = read_input_files(o2)[0][0]
        self.signal_time = np.arange(1, 257)
        self.rh, self.temp, self.pressure = conditions
        self.mask_locs = cuts

        self._get_mask()

        self._calc_h2o()
        self._calc_pathlength()
        self._calc_o2_conc()

    def _get_mask(self):
        first_cut, last_cut = self.mask_locs
        mask = [i for i in range(len(self.h2o)) if not first_cut < i < last_cut]
        self.mask = mask

    def _calc_h2o(self):
        ps = sat_vapour_pressure(self.temp)
        h2o = 10 ** 4 * self.rh * ps / 1013.25
        self.h2o_conc = h2o / 1E6

    def _process_signal(self, data, original):
        x = self.signal_time[self.mask]
        fit_params = np.polyfit(x, data, 7)
        x_poly = np.linspace(min(x), max(x), num=len(self.signal_time), endpoint=True)
        poly_fit = np.polyval(fit_params, x_poly)

        absorbance_profile = [-np.log(i / j) for i, j in zip(original, poly_fit)]
        transmittance_profile = [(1 - i) for i in absorbance_profile]
        transmittance_peak = min(transmittance_profile)
        absorbance_peak = max(absorbance_profile)
        return transmittance_peak

    def _calc_pathlength(self):
        gamma_v, s = 2.98E9, 1.936E-11

        h2o_masked = self.h2o[self.mask]
        peak = self._process_signal(h2o_masked, self.h2o)

        pathlength_measured = pathlength(self.h2o_conc, gamma_v, s, peak, self.temp, self.pressure)
        print(pathlength_measured)
        self.h2o_pathlength = pathlength_measured

    def _calc_o2_conc(self):
        gamma_v, s = 1.59E9, 2.36E-13

        o2_masked = self.o2[self.mask]
        peak = self._process_signal(o2_masked, self.o2)

        o2_measured = o2_concentration(self.h2o_pathlength, gamma_v, s, peak, self.temp, self.pressure)
        print(o2_measured)


def read_input_files(file):
    data, time, _ = read_input_file(file)
    return data, time


def arden_buck(t, p):
    p_part = 1.0007 + (3.46E-6 * p)
    p = p_part * 6.1121 * np.exp((18.678 - t / 234.5) * t / (257.14 + t))
    return p


def sat_vapour_pressure(t_atm):
    t = 1 - 373.15 / t_atm
    ps = 1013.25 * exp(13.3185 * t - 1.976 * t ** 2 - 0.6445 * t ** 3 - 0.1299 * t ** 4)
    return ps


def pathlength(h2o_conc, gamma_v, s, peak, temp, pressure):
    """
    pathlength = ln(T) / -SgN
    """
    boltzmann = constants.Boltzmann * 1E7
    N_0 = pressure / (boltzmann * temp)

    n = h2o_conc * N_0
    g_max = 1 / (np.pi * gamma_v)

    exponent = - s * g_max * n
    length = np.log(peak) / exponent
    return length


def o2_concentration(h2o_pathlength, gamma_v, s, peak, temp, pressure):
    """
    conc = ln(T) / -SgL
    """
    boltzmann = constants.Boltzmann * 1E7
    N_0 = pressure / (boltzmann * temp)
    N_0 = 1E6 / (boltzmann * temp)

    g_max = 1 / (np.pi * gamma_v)

    exponent = - s * g_max * h2o_pathlength
    conc = np.log(peak) / exponent
    conc = conc / N_0
    return conc


if __name__ == '__main__':
    calibration_h2o = 'input_gasmas/measurements/water_vapour/DAS_air_100_data.txt'
    calibration_o2 = 'input_gasmas/measurements/oxygen/DAS_oxy_100_data.txt'
    conditions_cal = [58.76, 19.3 + 273, 1020040]
    cuts_cal = [92, 160]
    check = CalibrationTest(calibration_h2o, calibration_o2, conditions_cal, cuts_cal)
