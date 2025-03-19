import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import os
import json
from scipy.optimize import least_squares, OptimizeResult

from data_processing.camera.process_thermal_images import deposition_rate
from data_processing.utils import get_experiment_params
import platform

DRIVE_PATH = r"/Users/erickmartinez/Library/CloudStorage/OneDrive-Personal"
if platform.system() == 'Windows':
    drive_path = r"C:\Users\erick\OneDrive"

BASE_DIR = r'Documents/ucsd/Postdoc/research/conferences/DPP 2023/figures'
DATABASE_CSV = 'Transmission measurements - thicknesses - 20250124.csv'

CSV_SOOT_DEPOSITION = r'Documents/ucsd/Postdoc/research/data/firing_tests' \
                      r'/surface_temperature/equilibrium_redone/slide_transmission_smausz_error.csv'

LASER_POWER_CSV = r'Documents/ucsd/Postdoc/research/data/firing_tests/LASER_POWER_MAPPING/laser_power_mapping.csv'

samples = [
    {'sample_id': 'R4N64', 'label': 'GC,  7.5% binder', 'material': 'Glassy carbon', 'marker': 'o', 'size': '850 um',
     'fig2': False, 'c': 'C0'},
    {'sample_id': 'R4N75', 'label': 'GC, 3.8% binder', 'material': 'Glassy carbon', 'marker': 'o', 'size': '850 um',
     'fig2': False, 'c': 'royalblue'},
    {'sample_id': 'R4N83', 'label': 'GC, 1.9% binder', 'material': 'Glassy carbon', 'marker': 'o', 'size': '850 um',
     'fig2': True, 'c': 'cornflowerblue'},
    # {'sample_id': 'R4N88', 'label': 'POCO spheres (1.0 mm)', 'material': 'POCO graphite', 'marker': 'h',
    #  'size': '850 um', 'fig2': True, 'c': 'C1'},
    # {'sample_id': 'R4N86', 'label': 'POCO cubes (1.7 mm)', 'material': 'POCO graphite', 'marker': 'v', 'size': '850 um',
    #  'fig2': True, 'c': 'brown'},
]


beam_radius = 0.5 * 0.8165  # * 1.5 # 0.707
n_cos, dn_cos = 6.5, 0.3
h_0, h_0_err = 10.5 * 2.54, 0.5 * 2.54

graphite_sample_diameter = 0.92
film_density = 2.2  # g / cm^3

def model_poly(x, b) -> np.ndarray:
    n = len(b)
    r = np.zeros(len(x))
    for i in range(n):
        r += b[i] * x ** i
    return r


def res_poly(b, x, y, w=1.):
    return (model_poly(x, b) - y) * w


def jac_poly(b, x, y, w=1):
    n = len(b)
    r = np.zeros((len(x), n))
    for i in range(n):
        r[:, i] = w * x ** i
    return r


eps = float(np.finfo(np.float64).eps)
def fit_polylog(xdata, ydata, yerror=None, weights=None, degree=5, loss='soft_l1', f_scale=1.0, tol=eps):
    if yerror is None:
        yerror = np.ones_like(xdata)
    if weights is None:
        weights = np.log(1 / (yerror + 0.1 * np.median(yerror)))
    fit_result_g = least_squares(
        res_poly, x0=[0.1 ** i for i in range(degree)], args=(xdata, np.log(ydata), weights),
        loss=loss, f_scale=f_scale,
        jac=jac_poly,
        xtol=tol,
        ftol=tol,
        gtol=tol,
        verbose=2,
        x_scale='jac',
        max_nfev=1000 * degree
    )
    return fit_result_g

def process_absolute_path(relative_path, drive_path):
    if platform.system() == 'Windows':
        relative_path = relative_path.replace("/", "\\")
    else:
        relative_path = relative_path.replace("\\", "/")
    abs_path = os.path.join(drive_path, relative_path)
    return abs_path


def gaussian_beam_aperture_factor(beam_radius, sample_radius):
    return 1.0 - np.exp(-2.0 * (sample_radius / beam_radius) ** 2.0)


sample_area = 0.25 * np.pi * (graphite_sample_diameter ** 2.)


def nmps2cps(deposit_rate):
    global film_density, h_0, n_cos, sample_area
    return 2. * np.pi * film_density * (h_0 ** 2.) * deposit_rate / n_cos / 12.011 * 6.02214076E-5 / sample_area # xE-20


def cps2nmps(deposit_rate):
    global film_density, h_0, n_cos, sample_area
    return 12.011 * n_cos / (2. * np.pi * film_density * (h_0 ** 2.)) * deposit_rate * sample_area / 6.02214076E-5

def nmps2tlpspm2(deposit_rate):
    global film_density, h_0, n_cos, sample_area
    return 2.1388E-03 * (h_0 ** 2.) * deposit_rate / n_cos /sample_area # Torr-L/s/m^2

def tlpspm22nmps(deposit_rate):
    global film_density, h_0, n_cos, sample_area
    return n_cos * sample_area * deposit_rate / 2.1388E-03 / (h_0 ** 2.)  # Torr-L/s/m^2


def map_laser_power_settings_from_files(base_dir, laser_power_dir):
    rdir = os.path.join(base_dir, laser_power_dir)
    file_list = os.listdir(rdir)
    mapping = {}
    for i, f in enumerate(file_list):
        if f.endswith('.csv'):
            params = get_experiment_params(relative_path=rdir, filename=os.path.splitext(f)[0])
            laser_setpoint = int(params['Laser power setpoint']['value'])
            df = pd.read_csv(os.path.join(rdir, f), comment='#').apply(pd.to_numeric)
            laser_power = df['Laser output peak power (W)'].values
            laser_power = laser_power[laser_power > 0.0]
            mapping[laser_setpoint] = laser_power.mean()

    keys = list(mapping.keys())
    keys.sort()
    return {i: mapping[i] for i in keys}

def map_laser_power_settings(laser_power_csv):
    df = pd.read_csv(laser_power_csv).apply(pd.to_numeric).sort_values(by=['Laser power setting (%)'], ascending=True)
    setpoint = df['Laser power setting (%)'].values
    laser_power = df['Laser power (W)'].values
    return {key: val for key, val in zip(setpoint, laser_power)}

def estimate_deposition_rate(
    film_thickness: np.ndarray, deposition_time: np.ndarray,
    film_thickness_error: np.ndarray, deposition_time_pct_error=0.1
):
    # Avoid division by zero by treating values of film thickness differently
    msk_zero = film_thickness == 0.
    deposition_rate = film_thickness / deposition_time
    deposition_rate[msk_zero] = film_thickness_error[msk_zero] / deposition_time[msk_zero] # minimum value based on uncertainty
    dt_by_t = np.ones_like(film_thickness) * deposition_time_pct_error
    dD_by_D = np.zeros_like(deposition_rate)
    dD_by_D[msk_zero] = 1. # Assume D = dD
    dD_by_D[~msk_zero] = film_thickness_error[~msk_zero] / film_thickness[~msk_zero]
    deposition_rate_error = deposition_rate * np.linalg.norm(
        np.column_stack([dt_by_t, dD_by_D]),
        axis=1
    )
    return deposition_rate, deposition_rate_error


def deposition_rate_to_sublimation_rate(
    deposition_rate, deposition_rate_error, h0, n, dh0, dn, area
):
    rs = deposition_rate * 2.1388E-03 * (h0 ** 2.) / n / area  * 1E4 #  Torr-L/s/m^2
    # Treat values with deposition_rate = 0 differently to avoid division by zero
    msk_zero = deposition_rate == 0
    rs[msk_zero] = deposition_rate_error[msk_zero] * 2.1388E-03 * (h0 ** 2.) / n / area * 1E4 # x10^4 Torr-L/s/m^2
    dn_by_n = dn / n * np.ones_like(deposition_rate)
    darea_by_area = 0.1 * np.ones_like(deposition_rate)
    drd_by_rd = np.zeros_like(deposition_rate)
    drd_by_rd[msk_zero] = 1 # If deposition rate (rd) = 0, assume rd=depostion_rate_error (drd)
    drd_by_rd[~msk_zero] = deposition_rate_error[~msk_zero] / deposition_rate[~msk_zero]
    dh0_by_h0 = 2. * (dh0 / h0) * np.ones_like(deposition_rate)
    drs = rs * np.linalg.norm(np.column_stack([drd_by_rd, dh0_by_h0, dn_by_n, darea_by_area]), axis=1)
    return rs, drs

def main(base_dir, database_csv, laser_power_csv, csv_soot_deposition, drive_path, h0, h0_err, ncos, dncos):
    base_dir = process_absolute_path(base_dir, drive_path)
    database_csv = os.path.join(base_dir, database_csv)
    laser_power_csv = process_absolute_path(laser_power_csv, drive_path)
    csv_soot_deposition = process_absolute_path(csv_soot_deposition, drive_path)
    df_main = pd.read_csv(
        os.path.join(base_dir, database_csv),
        # usecols=['Sample ID', 'Laser power setting (%)', 'Film thickness (nm)', 'Deposition rate (nm/s)']
    )
    numeric_columns = ['Laser power setting (%)', 'Film thickness (nm)', 'Deposition rate (nm/s)']
    df_main[numeric_columns] = df_main[numeric_columns].apply(pd.to_numeric)

    sample_id_list = [sid['sample_id'] for sid in samples]
    df = df_main[df_main['Sample ID'].isin(sample_id_list)]
    sample_ids = df['Sample ID'].unique()

    laser_power_mapping = map_laser_power_settings(laser_power_csv)


    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)

    # fig, axes = plt.subplots(ncols=1, nrows=2, gridspec_kw=dict(hspace=0, height_ratios=[2, 1.5]), sharex=True)
    fig, ax = plt.subplots(ncols=1, nrows=1, constrained_layout=True)
    fig.set_size_inches(4.0, 4.5)
    # fig.subplots_adjust(left=0.15, top=0.95, right=0.95, bottom=0.09)


    graphite_sample_diameter = 0.92
    graphite_area = 0.25 * np.pi * (graphite_sample_diameter ** 2.)

    pebble_sample_diameter = 1.025
    pebble_area = 0.25 * np.pi * (pebble_sample_diameter ** 2.)

    soot_deposition_df = pd.read_csv(csv_soot_deposition)
    soot_deposition_columns = soot_deposition_df.columns
    soot_deposition_df[soot_deposition_columns[1:]] = soot_deposition_df[soot_deposition_columns[1:]].apply(
        pd.to_numeric)
    soot_deposition_df = soot_deposition_df[soot_deposition_df['Flat top time (s)'] > 0]
    soot_deposition_pebble_df = soot_deposition_df[soot_deposition_df['Sample'] != 'GT001688']
    soot_deposition_graphite_df = soot_deposition_df[soot_deposition_df['Sample'] == 'GT001688']
    soot_deposition_pebble_df.sort_values(by=['Laser Power (%)'], ascending=True)
    soot_deposition_graphite_df.sort_values(by=['Laser Power (%)'], ascending=True)

    laser_power_setting_sublimation_graphite = soot_deposition_graphite_df['Laser Power (%)'].values
    laser_power_setting_sublimation_pebble = soot_deposition_pebble_df['Laser Power (%)'].values

    laser_power_graphite = np.array([laser_power_mapping[v] for v in laser_power_setting_sublimation_graphite])
    laser_power_pebble = np.array([laser_power_mapping[v] for v in laser_power_setting_sublimation_pebble])

    film_thickness_pebble = soot_deposition_pebble_df['Thickness (nm)'].values
    film_thickness_pebble_err = soot_deposition_pebble_df['Thickness error (nm)'].values
    flattop_time_pebble = soot_deposition_pebble_df['Flat top time (s)'].values
    depositon_rate_pebble, deposition_rate_pebble_err = estimate_deposition_rate(
        film_thickness=film_thickness_pebble, film_thickness_error=film_thickness_pebble_err,
        deposition_time=flattop_time_pebble
    )
    sublimation_rate_pebble, sublimation_rate_pebble_err = deposition_rate_to_sublimation_rate(
        deposition_rate=depositon_rate_pebble, deposition_rate_error=deposition_rate_pebble_err,
        h0=h0, dh0=h0_err, n=ncos, dn=dn_cos, area=pebble_area
    )


    film_thickness_graphite = soot_deposition_graphite_df['Thickness (nm)'].values
    film_thickness_graphite_err = soot_deposition_graphite_df['Thickness error (nm)'].values
    flattop_time_graphite = soot_deposition_graphite_df['Flat top time (s)'].values
    # msk_graphite = (film_thickness_graphite > 0.0) & (flattop_time_graphite > 0.0)
    msk_graphite = flattop_time_graphite > 0.0 # Avoid division by zero
    deposition_rate_graphite, deposition_rate_graphite_err = estimate_deposition_rate(
        film_thickness=film_thickness_graphite, film_thickness_error=film_thickness_graphite_err,
        deposition_time=flattop_time_graphite
    )
    sublimation_rate_graphite, sublimation_rate_graphite_err = deposition_rate_to_sublimation_rate(
        deposition_rate=deposition_rate_graphite, deposition_rate_error=deposition_rate_graphite_err,
        h0=h0, dh0=h0_err, n=ncos, dn=dn_cos, area=pebble_area
    )

    sample_area_graphite = 0.25 * np.pi * graphite_sample_diameter ** 2.0
    aperture_factor_graphite = gaussian_beam_aperture_factor(beam_radius=beam_radius,
                                                             sample_radius=0.5 * graphite_sample_diameter)
    incident_heat_load_graphite = aperture_factor_graphite * laser_power_graphite / sample_area_graphite / 100.0
    incident_heat_load_pebble = aperture_factor_graphite * laser_power_pebble / sample_area_graphite / 100.0

    evaporation_rate_graphite = 21.388E3 * (h_0 ** 2. / n_cos) * sublimation_rate_graphite


    heat_load_pebbles = incident_heat_load_pebble
    sublimation_rate_pebbles = sublimation_rate_pebble
    sublimation_rate_pebbles_err = sublimation_rate_pebble_err

    markers_p, caps_p, bars_p = ax.errorbar(
        incident_heat_load_pebble, sublimation_rate_pebble,  yerr=sublimation_rate_pebble_err,
        marker='o', ms=9, mew=1.25, mfc='none', label='Carbon pebble rods',
        capsize=2.75, elinewidth=1.25, lw=1.5, c='navy', ls='none'
    )
    [bar.set_alpha(0.35) for bar in bars_p]
    [cap.set_alpha(0.35) for cap in caps_p]


    for i, id in enumerate(sample_id_list):
        df2 = df[df['Sample ID'] == id]
        df2 = df2.groupby('Laser power setting (%)').agg({
            'Deposition rate (nm/s)': ['mean', 'std'],
            'Deposition rate lb (nm/s)': ['mean', 'std'],
            'Deposition rate ub (nm/s)': ['mean', 'std'],
            'Deposition rate error (nm/s)': ['mean', 'std'],
            'Evaporation rate (Torr-L/s)': ['mean', 'std'],
            'Evaporation rate lb (Torr-L/s)': ['mean', 'std'],
            'Evaporation rate ub (Torr-L/s)': ['mean', 'std'],
        })

        lbl = ""
        marker = samples[i]['marker']
        for s in samples:
            if s['sample_id'] == id:
                lbl = s['label']

        c = samples[i]['c']

        print(id, lbl)

        deposition_rate = df2['Deposition rate (nm/s)']['mean']
        deposition_rate_lb = df2['Deposition rate lb (nm/s)']['mean']
        deposition_rate_ub = df2['Deposition rate ub (nm/s)']['mean']
        deposition_rate_err = df2['Deposition rate error (nm/s)']['mean']

        sublimation_rate, sublimation_rate_err = deposition_rate_to_sublimation_rate(
            deposition_rate=deposition_rate, deposition_rate_error=deposition_rate_err,
            h0=h0, dh0=h0_err, n=ncos, dn=dn_cos, area=pebble_area
        )


        yerr_deposition = (nmps2tlpspm2(deposition_rate) - nmps2tlpspm2(deposition_rate_lb),
                           nmps2tlpspm2(deposition_rate_ub) - nmps2tlpspm2(deposition_rate))
        laser_power_setting = list(df2.index.values)
        laser_power = np.array([laser_power_mapping[v] for v in laser_power_setting])
        sample_diameter = 1.025
        sample_area = 0.25 * np.pi * sample_diameter ** 2.0
        aperture_factor = gaussian_beam_aperture_factor(beam_radius=beam_radius, sample_radius=0.5 * sample_diameter)
        incident_heat_load = aperture_factor * laser_power / sample_area / 100.0

        evaporation_rate_lb = df2['Evaporation rate lb (Torr-L/s)']['mean'] * 1E4 / sample_area
        evaporation_rate = df2['Evaporation rate (Torr-L/s)']['mean'] * 1E4 / sample_area
        evaporation_rate_ub = df2['Evaporation rate ub (Torr-L/s)']['mean'] * 1E4 / sample_area
        yerr_evaporation = (evaporation_rate - evaporation_rate_lb, evaporation_rate_ub - evaporation_rate_lb)


        markers_p, caps_p, bars_p = ax.errorbar(
            incident_heat_load, sublimation_rate, yerr=sublimation_rate_err,
            marker=marker, ms=9, mew=1.25, mfc='none', #label=f'{lbl}',
            capsize=2.75, elinewidth=1.25, lw=1.5, c=c, ls='none'
        )

        [bar.set_alpha(0.35) for bar in bars_p]
        [cap.set_alpha(0.35) for cap in caps_p]

        heat_load_pebbles = np.hstack((heat_load_pebbles, incident_heat_load))
        sublimation_rate_pebbles = np.hstack((sublimation_rate_pebbles, sublimation_rate))
        sublimation_rate_pebbles_err = np.hstack((sublimation_rate_pebbles_err, sublimation_rate_err))

    # secax = ax.secondary_yaxis('right', functions=(nmps2tlpspm2, tlpspm22nmps))
    # secax.set_ylabel(r'C/s/cm$^{\mathregular{2}}$')
    # secax.set_ylabel(r'x10$^{\mathregular{20}}$ C/s/cm$^{\mathregular{2}}$')
    # ax.set_ylabel(r'$\times$10$^{\mathregular{4}}$ [Torr-L/s/m$^{\mathregular{2}}$]')
    # secax.ticklabel_format(axis='y', useMathText=True)

    xpred = np.linspace(heat_load_pebbles.min(), heat_load_pebbles.max(), num=2000)

    rs_graphite_ub_log = np.log10(sublimation_rate_graphite + sublimation_rate_graphite_err)
    rs_graphite_lb_log = np.log10(sublimation_rate_graphite/rs_graphite_ub_log)
    rs_graphite_err_neg = sublimation_rate_graphite - 10. ** rs_graphite_lb_log

    fit_results_graphite = fit_polylog(
        xdata=incident_heat_load_graphite,
        ydata=sublimation_rate_graphite,
        yerror=sublimation_rate_graphite_err,
        f_scale=1E-1,
        loss='soft_l1',
        degree=3
    )


    markers_p, caps_p, bars_p = ax.errorbar(
        incident_heat_load_graphite, sublimation_rate_graphite,  yerr=(rs_graphite_err_neg,sublimation_rate_graphite_err),
        marker='D', ms=9, mew=1.25, mfc='none', label='Graphite rod',
        capsize=2.75, elinewidth=1.25, lw=1.5, c='tab:red', ls='none'
    )

    [bar.set_alpha(0.35) for bar in bars_p]
    [cap.set_alpha(0.35) for cap in caps_p]

    ax.plot(
        xpred, np.exp(model_poly(xpred, fit_results_graphite.x)),
        ls='--', c='tab:red'
    )

    # Sublimation at


    fit_results_pebbles = fit_polylog(
        xdata=heat_load_pebbles,
        ydata=sublimation_rate_pebbles,
        yerror=sublimation_rate_pebbles_err,
        f_scale=1E-1,
        loss='soft_l1',
        degree=3
    )



    ax.plot(
        xpred, np.exp(model_poly(xpred, fit_results_pebbles.x)),
        ls='--', c='tab:blue'
    )

    hl_boron = 5 * np.arange(1,9)

    # Boron measurements were taken with the lock-in amplifier which gave a std in the range of 500 uV instead
    # instead of 2 uV
    adjust_uncertainty_factor = 400E-6 / 0.02
    rs_boron = np.ones_like(hl_boron) * np.min(sublimation_rate_pebbles_err) * adjust_uncertainty_factor
    rs_boron_error = np.ones_like(rs_boron) * np.min(sublimation_rate_pebbles_err) * adjust_uncertainty_factor
    rs_boron_ub_log = np.log10(rs_boron + rs_boron_error)
    rs_boron_lb_log = np.log10(rs_boron_ub_log/rs_boron)
    rs_boron_err_neg = rs_boron - 10 ** rs_boron_lb_log
    markers_p, caps_p, bars_p = ax.errorbar(
        hl_boron, rs_boron, yerr=(rs_boron_err_neg,rs_boron_error),
        marker='^', ms=9, mew=1.25, mfc='C2', label='Boron pebble rod',
        capsize=2.75, elinewidth=1.25, lw=1.5, c='C2'
    )
    print(f"Mean sublimation rate err: {rs_boron.mean():.3E} ")

    [bar.set_alpha(0.35) for bar in bars_p]
    [cap.set_alpha(0.35) for cap in caps_p]

    # ax.set_xlabel('Heat load [MW/m$^{\mathregular{2}}$]')

    ax.set_title('Outgassing due to sublimation')
    ax.set_xlim(5, 40)
    ax.tick_params(which='both', axis='y', labelright=False, right=True, direction='out')
    ax.tick_params(which='both', axis='x', direction='out')

    ax.set_xlabel('Heat load (MW/m$^{\mathregular{2}}$)')


    ax.tick_params(which='both', axis='x', direction='out')

    ax.xaxis.set_major_locator(ticker.MultipleLocator(10.0))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5.0))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10.))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(5.))

    ax.set_yscale('log')
    ax.set_ylim(1E1, 1E6)
    ax.set_ylabel(r'Rate (Torr-L/s/m$^{\mathregular{2}}$)', fontweight='regular', fontsize=12)

    ax.axhline(y=1E3, ls='--', color='k')

    connectionstyle = "angle,angleA=-90,angleB=180,rad=0"
    bbox = dict(boxstyle="round", fc="wheat")
    arrowprops = dict(
        arrowstyle="->", color="k",
        shrinkA=5, shrinkB=0,
        patchA=None, patchB=None,
        # connectionstyle=connectionstyle
    )
    ax.annotate(
        f"Low-Z limit",
        xy=(35, 1E3), xycoords='data',  # 'figure pixels', #data',
        # transform=axes[1].transAxes,
        xytext=(0, 40), textcoords='offset pixels',
        ha='center', va='bottom',
        fontsize=11,
        arrowprops=arrowprops,
        bbox=bbox,
    )

    # W melting at 32 MW/m^2
    ax.axvline(x=32, ls='-.', lw=1.25, )

    ax.legend(loc='upper left', frameon=True, fontsize=11)

    # fig.tight_layout()

    file_tag = 'carbon_deposition_vs_laser_power_binder_content_20231020'

    fig.savefig(r'./figures/fig_emission_rates.svg', dpi=600)
    fig.savefig(r'./figures/fig_emission_rates.png', dpi=600)
    fig.savefig(r'./figures/fig_emission_rates.pdf', dpi=600)

    # fig2.savefig(os.path.join(base_dir, 'carbon_deposition_vs_laser_power_material_20231020' + '.png'), dpi=600)

    plt.show()


if __name__ == '__main__':
    main(
        base_dir=BASE_DIR, database_csv=DATABASE_CSV, laser_power_csv=LASER_POWER_CSV,
        csv_soot_deposition=CSV_SOOT_DEPOSITION, drive_path=DRIVE_PATH, h0=h_0, h0_err=h_0_err, ncos=n_cos, dncos=dn_cos
    )
