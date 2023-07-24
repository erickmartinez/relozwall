import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import json
import pandas as pd
import matplotlib.ticker as ticker
from scipy.optimize import least_squares
import data_processing.confidence as cf
from data_processing.utils import lighten_color
import ir_thermography.thermometry as irt
from data_processing.utils import get_experiment_params, latex_float


base_dir = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\thermal camera\calibration\calibration_20230721'
csv = 'adc_calibration_20230721.csv'
thermometry_csv = 'LT_GRAPHITE_100PCT_2023-07-19_2.csv'

cutoff_b = 7.75E-3

emissivity = 0.8



def model_poly(x, b):
    n = len(b)
    r = 0.
    xx = np.ones_like(x)
    for i in range(n):
        r += xx * b[i]
        xx = xx * x
    return r

def model_root(x, b):
    return b[0] * np.power(x, b[1]) + b[2]

def fobj_root(b, x, y):
    return model_poly(x, b) - y

def jac_root(b, x, y):
    n = len(b)
    j = np.ones((len(x), n), dtype=np.float64)
    p = np.power(x, b[1])
    j[:, 0] = p
    j[:, 1] = b[0] * p * np.log(b[1])
    return j

def fobj_poly(b, x, y):
    return model_poly(x, b) - y


def jac_poly(b, x, y):
    n = len(b)
    j = np.ones((len(x), n), dtype=np.float64)
    xx =x.copy()
    for i in range(1, n):
        j[:, i] = xx
        xx = xx * xx
    return j


def main():
    df = pd.read_csv(os.path.join(base_dir, csv)).apply(pd.to_numeric)
    adc_value = df['Average ADC'].values
    brightness_pd = df['Brightness at 900 mm (W/ster/cm^2)'].values
    temperature = df['Temperature (°C)'].values
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    idx_cutoff = brightness_pd <= cutoff_b
    adc_fit = adc_value[idx_cutoff]
    brightness_fit = brightness_pd[idx_cutoff]

    idx_temp_cutoff = (brightness_pd <= cutoff_b) & (temperature > 1000)
    adc_fit_temp = adc_value[idx_temp_cutoff]
    temperature_fit = temperature[idx_temp_cutoff]

    all_tol = np.finfo(np.float64).eps
    n = len(adc_value)
    b0 = np.array([brightness_fit[0], 0.1])
    res1 = least_squares(
        fobj_poly,
        b0,
        loss='cauchy', f_scale=0.001,
        jac=jac_poly,
        args=(adc_fit, brightness_fit),
        # bounds=([0., 0., 0., 0.], [brightness_fit.max(), np.inf, np.inf, np.inf]),
        xtol=all_tol,  # ** 0.5,
        ftol=all_tol,  # ** 0.5,
        gtol=all_tol,  # ** 0.5,
        max_nfev=10000 * n,
        # x_scale='jac',
        verbose=2
    )

    colors = ['C0', 'C1']

    popt1 = res1.x
    pcov1 = cf.get_pcov(res1)
    ci = cf.confint(n=n, pars=popt1, pcov=pcov1)
    xpred = np.linspace(adc_value.min(), adc_value.max(), 500)
    ypred1, lpb1, upb1 = cf.predint(x=xpred, xd=adc_fit, yd=brightness_fit, func=model_poly, res=res1)

    # b0 = np.array([temperature_fit.min(), 0.01, 0.])
    # res2 = least_squares(
    #     fobj_poly,
    #     b0,
    #     loss='soft_l1', f_scale=0.1,
    #     jac=jac_poly,
    #     args=(adc_fit_temp, temperature_fit),
    #     # bounds=([0., 0., 0.], [temperature_fit.max(), np.inf, np.inf]),
    #     xtol=all_tol,  # ** 0.5,
    #     ftol=all_tol,  # ** 0.5,
    #     gtol=all_tol,  # ** 0.5,
    #     max_nfev=100000 * n,
    #     # x_scale='jac',
    #     verbose=2
    # )
    #
    # colors = ['C0', 'C1']
    #
    # popt2 = res2.x
    # pcov2 = cf.get_pcov(res2)
    # ci = cf.confint(n=n, pars=popt2, pcov=pcov2)
    # ypred2, lpb2, upb2 = cf.predint(x=xpred, xd=adc_fit_temp, yd=temperature_fit, func=model_poly, res=res2)
    experiment_params = get_experiment_params(relative_path=base_dir, filename=os.path.splitext(thermometry_csv)[0])
    photodiode_gain = float(experiment_params['Photodiode Gain']['value'])
    thermometry = irt.PDThermometer()
    thermometry.gain = photodiode_gain
    thermometry.emissivity = emissivity
    inv_emissivity = 1. / emissivity
    ypred2 = thermometry.get_temperature_at_brightness(brightness=ypred1*inv_emissivity) - 273.15
    lpb2 = thermometry.get_temperature_at_brightness(brightness=lpb1*inv_emissivity) - 273.15
    upb2 = thermometry.get_temperature_at_brightness(brightness=upb1*inv_emissivity) - 273.15

    adc_full_scale = np.arange(0, 256)
    calibrated_brightness = model_poly(x=adc_full_scale, b=popt1)
    calibrated_temp = thermometry.get_temperature_at_brightness(brightness=calibrated_brightness*inv_emissivity) - 273.15


    fig, axes = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
    fig.set_size_inches(4.5, 6.0)
    axes[0].fill_between(xpred, lpb1, upb1, color=lighten_color(colors[0], 0.5))
    axes[0].plot(adc_value, brightness_pd, label='Brightness', marker='o', mfc='none', ls='none', c=colors[0])
    axes[0].plot(xpred, ypred1, color=lighten_color(colors[0], 1.5), label='Model')

    axes[1].fill_between(xpred, lpb2, upb2, color=lighten_color(colors[1], 0.5))
    axes[1].plot(adc_value, temperature, label='Brightness', marker='s', mfc='none', ls='none', c=colors[1])
    axes[1].plot(xpred, ypred2, color=lighten_color(colors[1], 1.5), label='Model')

    axes[0].set_yscale('log')

    axes[0].set_ylabel('Brightness (W/ster/nm/cm$^{\\mathregular{2}}$)')
    axes[0].set_ylim(1E-3, 1E-1)
    # axes[0].ticklabel_format(style='sci', axis='y', scilimits=(-2, 2), useMathText=True)
    axes[0].yaxis.set_major_locator(ticker.MultipleLocator(0.01))
    axes[0].yaxis.set_minor_locator(ticker.MultipleLocator(0.005))
    axes[0].set_title('Brightness at 900 nm')

    axes[1].set_ylabel('Temperature (°C)')
    axes[1].set_ylim(1000, 2500)
    axes[1].yaxis.set_major_locator(ticker.MultipleLocator(500))
    axes[1].yaxis.set_minor_locator(ticker.MultipleLocator(100))
    axes[1].set_title('Temperature (photodiode)')

    for ax in axes:
        ax.set_xlabel('ADC value')
        ax.set_xlim(0, 255)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
        ax.legend(loc='best', frameon=True, fontsize=9)

    fig.savefig(os.path.join(base_dir, 'adc_calibration.png'), dpi=300)

    fig2, ax2 = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig2.set_size_inches(4., 3.0)
    ax2.plot(adc_full_scale, calibrated_temp, color='tab:red')
    ax2.set_xlabel('ADC value')
    ax2.set_ylabel('Temperature (°C)')
    ax2.set_ylim(1000, 3000)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(500))
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(100))
    ax2.set_xlabel('ADC value')
    ax2.set_xlim(0, 255)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax2.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax2.set_title('Camera sensor calibration')

    fig2.savefig(os.path.join(base_dir, 'adc_calibration_curve.png'), dpi=300)

    camera_calibration_df = pd.DataFrame(data={
        'ADC value': adc_full_scale,
        'Temperature (°C)': calibrated_temp
    })

    camera_calibration_df.to_csv(
        path_or_buf=os.path.join(base_dir, 'adc_calibration_curve.csv'),
        index=False
    )

    plt.show()


if __name__ == '__main__':
    main()
