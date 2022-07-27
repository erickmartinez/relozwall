import numpy as np
import re
import os

def get_experiment_params(relative_path: str, filename: str):
    # Read the experiment parameters
    results_csv = os.path.join(relative_path, f'{filename}.csv')
    count = 0
    params = {}
    with open(results_csv) as f:
        for line in f:
            if line.startswith('#'):
                if count > 1:
                    l = line.strip()
                    print(l)
                    if l == '#Data:':
                        break
                    pattern1 = re.compile("\s+(.*?):\s(.*?)\s(.*?)$")
                    pattern2 = re.compile("\s+(.*?):\s(.*?)$")
                    matches1 = pattern1.findall(l)
                    matches2 = pattern2.findall(l)
                    if len(matches1) > 0:
                        params[matches1[0][0]] = {
                            'value': matches1[0][1],
                            'units': matches1[0][2]
                        }
                    elif len(matches2) > 0:
                        params[matches2[0][0]] = {
                            'value': matches2[0][1],
                            'units': ''
                        }
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
