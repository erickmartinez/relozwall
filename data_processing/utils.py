import numpy as np
import re
import os

import pandas as pd
from scipy.signal import savgol_filter


def get_experiment_params(relative_path: str, filename: str, debug=False):
    # Read the experiment parameters
    results_csv = os.path.join(relative_path, f'{filename}.csv')
    count = 0
    params = {}
    with open(results_csv) as f:
        for line in f:
            if line.startswith('#'):
                if count > 1:
                    l = line[1::].strip()
                    # l = line.strip()
                    if debug:
                        print(l)
                    if l == 'Data:':
                        break

                    # find the text separated by the colons
                    pattern_colon = re.compile(r"\s*([^:]*)")
                    # Try to find all occurrences and remove empty strings
                    m0 = pattern_colon.findall(l)
                    m1 = [m for m in m0 if len(m.strip()) > 0]
                    pattern_num_units = re.compile(r"\s*([-+]?\d+\.?\d*[eE]?\+?\-?\d*?)\s(.*?)$")
                    m2 = pattern_num_units.match(m1[1])

                    param_name = m1[0]
                    param_value = m1[1]
                    param_units = ''
                    if not m2 is None:
                        # if debug:
                        #     print(m2.groups())
                        param_value = m2.group(1)
                        param_units = m2.group(2)
                    params[param_name] = {
                        'value': param_value, 'units': param_units
                    }
                    #
                    # pattern1 = re.compile("\s+(.*?):\s(.*?)\s+(.*?)$")
                    # pattern2 = re.compile("\s+(.*?):\s(.*)$")
                    # matches1 = pattern1.findall(l)
                    # matches2 = pattern2.findall(l)
                    # if len(matches1) > 0:
                    #     params[matches1[0][0]] = {
                    #         'value': matches1[0][1],
                    #         'units': matches1[0][2]
                    #     }
                    # elif len(matches2) > 0:
                    #     params[matches2[0][0]] = {
                    #         'value': matches2[0][1],
                    #         'units': ''
                    #     }
                count += 1
    return params


def latex_float(f, significant_digits=2):
    # significant_digits += 1
    float_exp_str = f"{{val:.{significant_digits}e}}"
    float_float_str = f"{{val:.{significant_digits}f}}"
    use_exponential = 1E3 < abs(f) or abs(f) < 1E-2
    # print(f"Input value: {f}")
    # print(f"Use exponential: {use_exponential}")
    float_str = float_exp_str.format(val=f).lower() if use_exponential else float_float_str.format(val=f).lower()

    if "e" in float_str:
        base, exponent = float_str.split("e")
        # return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
        if exponent[0] == '+':
            exponent = exponent[1::]
        return rf"{base} \times 10^{{{int(exponent)}}}"
    else:
        return float_str


def latex_float_with_error(value, error=None, digits=2):
    value_exp_str = f"{{value:.{digits}e}}"
    value_float_str = f"{{value:.{digits}f}}"
    error_exp_str = f"{{value:.{digits}e}}"
    error_float_str = f"{{value:.{digits}f}}"
    use_exponential = 1E3 < abs(value) or abs(value) < 1E-2
    # print(f"Input value: {f}")
    # print(f"Use exponential: {use_exponential}")
    float_str = value_exp_str.format(value=value).lower() if use_exponential else value_float_str.format(
        value=value).lower()
    if error is not None:
        error_str = error_exp_str.format(value=error).lower() if use_exponential else error_float_str.format(
            value=error).lower()

    if "e" in float_str:
        v_base, v_exponent = float_str.split("e")

        # return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
        if v_exponent[0] == '+':
            v_exponent = v_exponent[1::]
        v_base_float, v_exponent_float = float(v_base), float(v_exponent)
        fs = f"{{v_base:.{digits}f}}"
        lf = fs.format(v_base=v_base_float)
        if error is not None:
            e_base, e_exponent = error_str.split("e")
            if e_exponent[0] == '+':
                e_exponent = e_exponent[1::]
            e_base_float, e_exponent_float = float(e_base), float(e_exponent)
            pef = 10. ** (v_exponent_float - e_exponent_float)
            fs = rf"({{v_base:.{digits}f}}\pm{{e_base:.{digits}f}})"
            lf = fs.format(v_base=v_base_float * pef, e_base=e_base_float)
        return rf"{lf} \times 10^{{{e_exponent_float:.0f}}}"
    else:
        return float_str + '\\pm' + error_str


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def correct_thermocouple_response(measured_temperature, measured_time, tau):
    n = len(measured_time)
    k = int(n / 15)
    k = k + 1 if k % 2 == 0 else k
    k = max(k, 5)
    # T = savgol_filter(measured_temperature, k, 3)
    # dTdt = np.gradient(T, measured_time, edge_order=2)
    delta = measured_time[1] - measured_time[0]
    dTdt = savgol_filter(x=measured_temperature, window_length=k, polyorder=4, deriv=1, delta=delta)
    # dTdt = savgol_filter(dTdt, k - 2, 3)
    r = measured_temperature + tau * dTdt
    return savgol_filter(r, k - 4, 3)


def specific_heat_of_graphite(temperature: float, **kwargs):
    units = kwargs.get('units', 'K')
    T = temperature if units == 'K' else (temperature + 273.15)
    cp = 0.538657 + 9.11129E-6 * T - 90.2725 * (T ** (-1)) - 43449.3 * (T ** (-2.0)) + 1.59309E7 * (
            T ** (-3.0)) - 1.43688E9 * (T ** (-4.0))
    return cp * 4.184


LASER_POWER_MAPPING = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\LASER_POWER_MAPPING' \
                      r'\laser_power_mapping.csv'


def get_laser_power_mapping(csv=LASER_POWER_MAPPING):
    df = pd.read_csv(csv).apply(pd.to_numeric)
    mapping = {}
    for i, r in df.iterrows():
        mapping[int(r['Laser power setting (%)'])] = r['Laser power (W)']

    keys = list(mapping.keys())
    keys.sort()
    return {i: mapping[i] for i in keys}
