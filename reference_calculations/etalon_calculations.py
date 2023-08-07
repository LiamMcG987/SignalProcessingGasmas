import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from utils import read_input_file


class FrequencyCalculations:

    def __init__(self, input_file):
        self.input_file = input_file
        self._read_input_file()
        self._visualise_inputs()
        self._locate_peaks()
        self._calc_fsr()

    def _read_input_file(self):
        self.signal_wms, self.signal_time_wms = read_input_file(self.input_file, wms=True)
        self.signal_das, self.signal_time_das = read_input_file(self.input_file)

    def _visualise_inputs(self):
        fig, ax = plt.subplots()

        plt.title('Etalon - Frequency')

        ax.plot(self.signal_time_wms, self.signal_wms,
                label='WMS', color='red')
        ax.set_xlabel('Relative Frequency')
        ax.set_ylabel('Signal Strength')

        ax_2 = ax.twinx()
        ax_2.plot(self.signal_time_das, self.signal_das,
                  label='DAS', color='green')
        ax_2.set_ylabel('Harmonic Amplitude')
        fig.legend()
        plt.show()

    def _locate_peaks(self):
        peak_locs, _ = signal.find_peaks(self.signal_wms)
        peaks = np.asarray(self.signal_wms)[peak_locs]

        for i, peak in enumerate(peak_locs):
            if i == 0:
                continue

            previous_peak = peak_locs[i-1]
            time_diff = self.signal_time_wms[peak] - self.signal_time_wms[previous_peak]

    def _calc_fsr(self):
        """
        FSR = c / (2 * n * L)
        c = speed of light, n = refractive index, L = thickness.
        """
        S = 1.936E-11
        g = 7.4026E-11
        N = 5.2E17

        t_min = 0.9393923939948811

        L = calc_pathlength(t_min, S, g, N)
        print(L)


def calc_pathlength(t_min, s, g, n):
    pathlength = np.log(t_min) / (-s * g * n)
    return pathlength


if __name__ == '__main__':
    file = '../input_gasmas/etalon/Etalon_WMS_DAS_data_2023-05-12_155142.txt'
    FrequencyCalculations(file)
