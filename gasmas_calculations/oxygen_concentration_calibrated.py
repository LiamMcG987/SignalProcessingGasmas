from math import exp

from utils import beer_lambert


def calc_gasmas_signal(h20_params, o2_params, peaks, cal_constant, temperature, pressure, rh):
    gamma_h20, s_h20 = h20_params
    gamma_o2, s_o2 = o2_params
    trans_peak_h20, trans_peak_o2 = peaks
    abs_peak_o2 = 1 - trans_peak_o2

    h20_conc = calc_h20_conc(temperature, rh, pressure)

    gasmas_h20_pathlength = beer_lambert('pathlength', gamma_h20, s_h20,
                                         trans_peak_h20, temperature, pressure,
                                         c=h20_conc)
    # gasmas_o2_conc = beer_lambert('concentration', gamma_o2, s_o2,
    #                               trans_peak_o2, temperature, pressure,
    #                               pathlength=gasmas_h20_pathlength)
    print(f'pathlength: {gasmas_h20_pathlength}')

    o2_conc_calibrated = gasmas_o2_calc(abs_peak_o2, cal_constant, gasmas_h20_pathlength)
    print(f'concentration calibrated: {o2_conc_calibrated}')
    return o2_conc_calibrated


def gasmas_o2_calc(abs_peak, constant, pathlength):
    gasmas_signal = abs_peak * constant / pathlength
    return gasmas_signal


def calc_h20_conc(t_atm, rh, pressure):
    """
    h20 = 10**4 * rh * Ps(Ta)/Pa    (ppmv).
    """
    ps = sat_vapour_pressure(t_atm)

    h2o = 10 ** 4 * rh * ps / pressure
    return h2o / 1E6


def sat_vapour_pressure(t_atm):
    t = 1 - 373.15 / t_atm
    ps = 1013.25 * exp(13.3185 * t - 1.976 * t ** 2 - 0.6445 * t ** 3 - 0.1299 * t ** 4)
    return ps
