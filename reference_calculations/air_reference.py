import numpy as np
from math import exp

from utils import read_input_file, beer_lambert
from gasmas_calculations.gasmas_signal import GasmasSignalProcessing


class AirReference:

    def __init__(self, input_file_h2o, input_file_o2, t_atm, rh, pressure):
        self.input_file_h2o = input_file_h2o
        self.input_file_o2 = input_file_o2
        self.t_atm = t_atm
        self.rh = rh
        self.pressure = pressure

        self._read_input_files()
        self._calc_h20_conc()
        self._calc_pathlength()

        self._calc_o2_conc()

    def _read_input_files(self):
        self.signal_h2o, self.time_h2o, self.pathlengths_h2o = read_input_file(self.input_file_h2o)
        self.signal_o2, self.time_o2, self.pathlengths_o2 = read_input_file(self.input_file_o2)

    def _calc_h20_conc(self):
        """
        h20 = 10**4 * rh * Ps(Ta)/Pa    (ppmv).
        """
        ps = arden_buck(self.t_atm, self.pressure)
        h2o = 10 ** 4 * self.rh * ps / self.pressure
        self.h2o = h2o

    def _calc_pathlength(self):
        """
        b = G_w/h20
        L_eq = L_gas * c_gas/c_ref
        """
        signals = self.signal_h2o
        conc_eff = self.h2o * 1E-6
        gamma_v, s = 2.98E9, 1.936E-11

        pathlengths_calc_h2o = np.zeros(len(signals))
        print('## water vapour ##')

        for i, signal in enumerate(signals):
            gasmas = GasmasSignalProcessing(signal, plots=False)
            transmittance_peak = gasmas.transmittance_peak

            pathlength = beer_lambert('pathlength', gamma_v, s, transmittance_peak, self.t_atm, self.pressure,
                                      c=conc_eff)
            print(f'peak (trans): {round(transmittance_peak, 4)}')
            pathlengths_calc_h2o[i] = pathlength + 0.9

        print(f'\npathlengths: {pathlengths_calc_h2o}')
        self.pathlengths_calc_h2o = pathlengths_calc_h2o

    def _calc_o2_conc(self):
        signals = self.signal_o2
        pathlengths_calculated = self.pathlengths_calc_h2o
        gamma_v, s = 1.59E9, 2.36E-13

        o2_conc_calc = np.zeros(len(signals))
        calibration_constants = np.zeros(len(signals))
        print('## oxygen ##')

        for i, signal in enumerate(signals):
            gasmas = GasmasSignalProcessing(signal, plots=False)
            transmittance_peak = gasmas.transmittance_peak
            absorbance_peak = 1 - transmittance_peak
            print(f'peak (trans): {round(transmittance_peak, 4)}')

            pathlength = pathlengths_calculated[i]
            o2_conc = beer_lambert('concentration', gamma_v, s, transmittance_peak, self.t_atm, self.pressure,
                                   pathlength=pathlength)
            o2_conc_calc[i] = o2_conc
            calibration_constants[i] = o2_conc * pathlength / absorbance_peak
        print(f'\noxygen concentrations: {o2_conc_calc * 100}')
        print(f'calibration constants: {calibration_constants}')
        # breakpoint()
        self.calibration_constant_avg = np.mean(calibration_constants)


def calc_hwhm(absorbance, peak):
    abs_peak = - np.log(peak)
    half_maximum = abs_peak / 2

    idx = np.argwhere(np.diff(np.sign(absorbance - half_maximum))).flatten()
    hw = np.diff(idx) / 2
    hw = hw/len(absorbance)
    hwhm = 1 / (hw * 3.085E-9)

    return hwhm


def sat_vapour_pressure(t_atm):
    t = 1 - 373.15 / t_atm
    ps = 1013.25 * exp(13.3185 * t - 1.976 * t ** 2 - 0.6445 * t ** 3 - 0.1299 * t ** 4)
    return ps


def arden_buck(t, p):
    p_part = 1.0007 + (3.46E-6 * p)
    p = p_part * 6.1121 * np.exp((18.678 - t / 234.5) * t / (257.14 + t))
    return p
