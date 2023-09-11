import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, signal, fft
from sklearn.metrics import r2_score
import itertools

from utils import read_input_file


class GasmasSignalProcessing:

    def __init__(self, input_file,
                 known_cuts=None, known_poly_order=None, plots=None):

        self.known_cuts = known_cuts if known_cuts is not None else False
        self.known_poly_order = known_poly_order if known_poly_order is not None else False
        self.plots = plots if plots is not None else True

        self.input_file = input_file

        self.gamma_v_o2 = 1.59E9
        self.s_o2 = 2.36E-13

        self.gamma_v_h20 = 2.98E9
        self.s_h20 = 1.936E-11

        self._read_input_file()

        if not self.known_cuts:
            self._sg_preprocessing()
        else:
            self._choose_cuts()

        self._apply_mask()
        if not self.known_poly_order:
            self._fit_poly()
        else:
            self._fit_poly_known()

    def _read_input_file(self):
        if isinstance(self.input_file, (list, np.ndarray)):
            self.input_signal, self.signal_time = self.input_file, np.arange(1, len(self.input_file) + 1)
        else:
            self.input_signal, self.signal_time, _ = read_input_file(self.input_file)

    def _sg_preprocessing(self):
        sg_fit = np.asarray(signal.savgol_filter(self.input_signal, 10, 4, deriv=2))
        sg_max = max(sg_fit)

        if sg_max in sg_fit[:5] or sg_max in sg_fit[-5:]:
            sg_max = max(sg_fit[5:-5])

        sg_max_idx = [i for i, x in enumerate(sg_fit) if x == sg_max]

        peak_locs_max, _max = signal.find_peaks(sg_fit, prominence=1)
        peak_locs_min, _min = signal.find_peaks(-sg_fit, prominence=1)

        peak_locs = np.asarray(sorted(list(itertools.chain(peak_locs_min, peak_locs_max))))
        peaks = sg_fit[peak_locs]

        sg_max_idx_list = np.where(peak_locs == sg_max_idx)[0]
        peak_region = np.arange(peak_locs[sg_max_idx_list - 2], peak_locs[sg_max_idx_list + 4] + 1)

        self.peak_region = peak_region
        self.first_cut = peak_region[0]
        self.last_cut = peak_region[-1]
        print(f'optimal cut locations at {self.first_cut} and {self.last_cut}.')

        if self.plots:
            fig, ax = plt.subplots(2, 1)
            ax[0].plot(sg_fit, label='sg')
            ax[0].scatter(peak_locs, peaks)
            ax[0].scatter(peak_region, sg_fit[peak_region])
            ax[1].plot(self.input_signal, label='input')
            ax[1].axvline(peak_region[0], color='red', linestyle='--')
            ax[1].axvline(peak_region[-1], color='red', linestyle='--')
            plt.legend()
            plt.show()

    def _choose_cuts(self):
        cuts = self.known_cuts
        first_cut, last_cut = cuts[0], cuts[1]
        self.peak_region = np.arange(first_cut, last_cut + 1)
        self.first_cut = first_cut
        self.last_cut = last_cut

    def _apply_mask(self):
        mask = [i for i, j in enumerate(self.signal_time) if not self.first_cut < j < self.last_cut]
        self.excluded_range = mask

    def _fit_poly_known(self):
        poly_order = self.known_poly_order
        x = self.signal_time[self.excluded_range]
        y = self.input_signal[self.excluded_range]

        fit_params = np.polyfit(x, y, poly_order)
        x_poly = np.linspace(min(x), max(x), num=len(self.signal_time), endpoint=True)
        poly_fit = np.polyval(fit_params, x_poly)
        self._calc_transmittance(poly_fit, poly_order)

    def _fit_poly(self):
        """
        add hwhm filter
        """
        poly_degrees = np.arange(3, 8)
        diffs = np.zeros(shape=(len(poly_degrees), 2))
        poly_fits = np.zeros(shape=(len(poly_degrees), len(self.signal_time)))

        x = self.signal_time[self.excluded_range]
        y = self.input_signal[self.excluded_range]

        for i, degree in enumerate(poly_degrees):
            fit_params = np.polyfit(x, y, degree)
            x_poly = np.linspace(min(x), max(x), num=len(self.signal_time), endpoint=True)
            poly_fit = np.polyval(fit_params, x_poly)
            poly_fits[i] = poly_fit

            diffs[i][0] = degree
            diffs[i][1] = self._calc_transmittance(poly_fit, degree, iterate=True)

        poly_order_opt = get_idx(diffs, max)
        opt_idx = np.where(diffs[:, 0] == poly_order_opt)[0]
        poly_fit_opt = poly_fits[opt_idx][0]
        print(f'optimal polynomial order is {poly_order_opt}.')
        self.poly_order_opt = poly_order_opt
        self._calc_transmittance(poly_fit_opt, poly_order_opt)

    def _calc_transmittance(self, poly, degree, iterate=False):
        x = self.signal_time
        transmittance = [i / j for i, j in zip(self.input_signal, poly)]
        absorbance = [-np.log(i) for i in transmittance]

        self.absorbance_profile = absorbance
        self.transmittance_profile = transmittance
        self.absorbance_peak = max(absorbance)
        self.transmittance_peak = 1 - self.absorbance_peak

        func = lorentzian_func
        if iterate:
            return fit_to_func(x, absorbance, func, iterate=iterate, order=degree, plots=self.plots)
        else:
            fit_to_func(x, absorbance, func, order=degree, plots=self.plots)


def fit_to_func(x, y, func, iterate=False, order=None, plots=True):
    p0 = [0.0035, 121, 11]

    parameters, covariance = optimize.curve_fit(func, x, y, p0=p0)

    fit = func(x, *parameters)
    r2 = r2_score(y, fit)

    fit_vs_measured = []
    for i in range(len(fit)):
        diff = y[i] - fit[i]
        fit_vs_measured.append(diff)

    if iterate:
        return r2

    if plots:
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(x, y, '-.', label='measured')
        ax[0].plot(x, fit, '-', label='fitted')
        ax[0].set_ylabel('Absorbance (a.u)')
        ax[0].legend(loc='upper right')

        ax[1].plot(x, fit_vs_measured, label='difference', linestyle='--')
        ax[1].axhline(0, color='red', linestyle='-.', alpha=0.2, label='zero line')
        ax[1].set_ylabel('Magnitude (a.u)')
        ax[1].set_xlabel('Relative Frequency (GHz)')
        ax[1].legend(loc='upper right')

        fig.suptitle(f'Fitted Curve. Polynomial Order: {order}')
        plt.tight_layout()
        plt.show()


def lorentzian_func(x, p1, p2, p3):
    y = p1 / (((x - p2) / p3) ** 2 + 1)
    return y


def get_idx(diffs, kwarg, two_d=False):
    m_diff = kwarg(np.asarray(diffs)[:, -1])
    m_diff_idx_loc = np.where(np.asarray(diffs)[:, -1] == m_diff)
    if two_d:
        cut_opt = np.asarray(diffs)[m_diff_idx_loc]
    else:
        cut_opt = int(np.asarray(diffs)[m_diff_idx_loc][0][0])

    return cut_opt


def gasmas_o2_calc(abs_peak, constant, pathlength):
    gasmas_signal = abs_peak * constant / pathlength
    return gasmas_signal
