import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import json
from scipy.interpolate import interp1d
import data_processing.echelle as ech
import os
from scipy.stats.distributions import t
from scipy.signal import savgol_filter

window_coefficients = np.array([12.783, 0.13065, -8.467e-5])

def load_db():
    df = pd.read_excel('./data/echelle_db.xlsx', sheet_name='Spectrometer parameters')
    df = df[df['Folder'] == 'echelle_20240910'].reset_index(drop=True)
    return df

def load_labsphere_calibration():
    df = pd.read_csv(
        './data/PALabsphere_2014.txt', sep=' ', comment='#',
        usecols=[0], names=['Radiance (W/cm2/ster/nm)']
    ).apply(pd.to_numeric)
    # radiance = df['Radiance (W/cm2/ster/nm)']
    n = len(df)
    wl = 350. + np.arange(n) * 10.
    df['Wavelength (nm)'] = wl
    return df[['Wavelength (nm)', 'Radiance (W/cm2/ster/nm)']]

def transmission_dirty_window(wavelength: np.ndarray) -> np.ndarray:
    global window_coefficients
    wavelength = np.array(wavelength)
    n = len(window_coefficients)
    m = len(wavelength)
    x = np.ones(m, dtype=np.float64)
    transmission = np.zeros(m, dtype=np.float64)
    for i in range(n):
        transmission += window_coefficients[i] * x
        x = x * wavelength
    return transmission

def load_plot_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)
    mpl.rcParams['text.latex.preamble'] = (r'\usepackage{mathptmx}'
                                           r'\usepackage{xcolor}'
                                           r'\usepackage{helvet}'
                                           r'\usepackage{siunitx}')

def main():
    db_df = load_db()
    labsphere_df = load_labsphere_calibration()
    wl_ls = labsphere_df['Wavelength (nm)'].values
    radiance_ls = labsphere_df['Radiance (W/cm2/ster/nm)'].values
    # For wavelengths below labsphere data, use the extrapolated data from the fit
    labsphere_extrapolated_csv = r'./data/PALabsphere_2014_extrapolated.txt'
    labsphere_extrapolated_df = pd.read_csv(labsphere_extrapolated_csv, comment='#').apply(pd.to_numeric)
    wl_ls_xp = labsphere_extrapolated_df['Wavelength (nm)'].values
    radiance_ls_xp = labsphere_extrapolated_df['Radiance (W/cm^2/ster/nm)'].values
    wl_ls = np.hstack([wl_ls_xp, wl_ls])
    radiance_ls = np.hstack([radiance_ls_xp, radiance_ls])
    # photon_energy_ev = 1239.84198433 * np.power(wl_ls, -1.)
    # photon_energy_J = 19.86445857E-17 * np.power(wl_ls, -1.)
    photon_flux_ls = 5.03411656E15 * wl_ls * radiance_ls
    # print(np.isclose(photon_flux_ls, radiance_ls / photon_energy_J, atol=1E-5))
    flux_ls_interp = interp1d(x=wl_ls, y=photon_flux_ls)
    radiance_ls_interp = interp1d(x=wl_ls, y=radiance_ls)




    # print(db_df)

    cmap = mpl.colormaps.get_cmap('jet_r')
    n_files = len(db_df)
    norm = mpl.colors.Normalize(vmin=0, vmax=(n_files-1))
    colors = [cmap(norm(i)) for i in range(n_files)]
    markers = ['o', 's', '^', 'v', 'D', '<', '>', 'h', 'd', '*', 'H', 'p']

    load_plot_style()
    fig_raw, axes = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
    fig_raw.set_size_inches(5.5, 5.5)
    # fig.subplots_adjust(hspace=0, left=0.15, right=0.95, bottom=0.125, top=0.95)

    wl_pa1_3300K = []
    wl_pa4_3300K = []
    wl_pa1_bgnd = []
    wl_pa4_bgnd = []
    counts_pa1_3300K = []
    counts_pa4_3300K = []
    counts_pa1_bgnd = []
    counts_pa4_bgnd = []

    # Iterate for every calibration
    for i, row in db_df.iterrows():
        path_to_echelle = os.path.join('./data/Echelle_data', row['Folder'], row['File'])
        df, params = ech.load_echelle_file(path_to_file=path_to_echelle)
        df = df[df['wl (nm)'].between(wl_ls.min(), wl_ls.max())].reset_index(drop=True)
        preamp_gain = row['Pre-Amplifier Gain']
        exposure_s = row['Exposure Time (secs)']
        accumulations = row['Number of Accumulations']
        wl_i = df['wl (nm)'].values
        # dwl = np.diff(wl_i)
        # print('d_wl.mean:', dwl.mean(), 'd_wl.min():', dwl.min(), 'd_wl.max:', dwl.max(), 'd_wl.std:', dwl.std())
        counts = df['counts'].values
        counts -= counts.min()
        # counts[counts < 0.] = 0.
        counts_ps = counts / exposure_s / accumulations
        # counts_ps = savgol_filter(
        #     counts_ps,
        #     window_length=53,
        #     polyorder=3
        # )

        if  row['Is dark'] == 0:
            if preamp_gain == 1:
                counts_pa1_3300K.append(counts_ps)
                wl_pa1_3300K.append(wl_i)
            elif preamp_gain == 4:
                counts_pa4_3300K.append(counts_ps)
                wl_pa4_3300K.append(wl_i)
        elif row['Is dark'] == 1:
            if preamp_gain == 1:
                counts_pa1_bgnd.append(counts_ps)
                wl_pa1_bgnd.append(wl_i)
            elif preamp_gain == 4:
                counts_pa4_bgnd.append(counts_ps)
                wl_pa4_bgnd.append(wl_i)
        if i < 6:
            axes[0].plot(
                wl_i, counts_ps, color=colors[i], label=f'Run {i + 1:>2d}, Preamp gain: {preamp_gain:>2.0f}',
                # marker=markers[i], markevery=500, mfc='none', ms=8, mew=1.5
            )
        else:
            axes[1].plot(
                wl_i, counts_ps, color=colors[i], label=f'Run {i + 1:>2d}, Preamp gain: {preamp_gain:>2.0f}',
                # marker=markers[i], markevery=500, mfc='none', ms=8, mew=1.5
            )

    # Average all the values and get the std
    wl_pa1_3300K = np.array(wl_pa1_3300K).T
    wl_pa4_3300K = np.array(wl_pa4_3300K).T
    wl_pa1_bgnd = np.array(wl_pa1_bgnd).T
    wl_pa4_bgnd = np.array(wl_pa4_bgnd).T

    counts_pa1_3300K = np.array(counts_pa1_3300K).T
    counts_pa4_3300K = np.array(counts_pa4_3300K).T
    counts_pa1_bgnd = np.array(counts_pa1_bgnd).T
    counts_pa4_bgnd = np.array(counts_pa4_bgnd).T

    wl_pa1_3300K_mean = wl_pa1_3300K.mean(axis=1)
    # It seems that the wavelength vectors is the same in all calibration files
    # wl_pa4_3300K_mean = wl_pa4_3300K.mean(axis=1)
    # wl_pa1_bgnd_mean = wl_pa1_bgnd.mean(axis=1)
    # wl_pa4_bgnd_mean = wl_pa4_bgnd.mean(axis=1)

    counts_pa1_3300K_mean = counts_pa1_3300K.mean(axis=1)
    counts_pa4_3300K_mean = counts_pa4_3300K.mean(axis=1)
    counts_pa1_bgnd_mean = counts_pa1_bgnd.mean(axis=1)
    counts_pa4_bgnd_mean = counts_pa4_bgnd.mean(axis=1)

    counts_pa1_3300K_mean = savgol_filter(
        counts_pa1_3300K_mean,
        window_length=93,
        polyorder=3
    )
    counts_pa4_3300K_mean = savgol_filter(
        counts_pa4_3300K_mean,
        window_length=93,
        polyorder=3
    )

    counts_pa1_bgnd_mean = savgol_filter(
        counts_pa1_bgnd_mean,
        window_length=93,
        polyorder=3
    )

    counts_pa4_bgnd_mean = savgol_filter(
        counts_pa4_bgnd_mean,
        window_length=93,
        polyorder=3
    )

    counts_pa1_3300K_mean -= counts_pa1_3300K_mean.min()
    counts_pa4_3300K_mean -= counts_pa4_3300K_mean.min()
    counts_pa1_bgnd_mean -= counts_pa1_bgnd_mean.min()
    counts_pa4_bgnd_mean -= counts_pa4_bgnd_mean.min()

    counts_pa1_3300K_std = counts_pa1_3300K.std(ddof=1, axis=1)
    counts_pa4_3300K_std = counts_pa4_3300K.std(ddof=1, axis=1)
    counts_pa1_bgnd_std = counts_pa1_bgnd.std(ddof=1, axis=1)
    counts_pa4_bgnd_std = counts_pa4_bgnd.std(ddof=1, axis=1)

    confidence_level = 0.95
    alpha = 1. - confidence_level
    tval = t.ppf(1 - 0.5 * alpha, counts_pa1_3300K.shape[1]-1)
    se_factor = tval / np.sqrt(counts_pa1_3300K.shape[1])

    counts_pa1_3300K_se = counts_pa1_3300K_std * se_factor
    counts_pa4_3300K_se = counts_pa4_3300K_std * se_factor
    counts_pa1_bgnd_se = counts_pa1_bgnd_std * se_factor
    counts_pa4_bgnd_se = counts_pa4_bgnd_std * se_factor


    # print('wl1 3300K equal to wl4 3300K?', np.isclose(wl_pa1_3300K_mean, wl_pa4_3300K_mean).all())
    # print('wl1 3300K equal to wl1 bgnd?', np.isclose(wl_pa1_3300K_mean, wl_pa1_bgnd_mean).all())
    # print('wl1 3300K equal to wl2 bgnd?', np.isclose(wl_pa1_3300K_mean, wl_pa4_bgnd_mean).all())

    delta_counts_pa1 = counts_pa1_3300K_mean - counts_pa1_bgnd_mean
    delta_counts_pa4 = counts_pa4_3300K_mean - counts_pa4_bgnd_mean

    # If the difference between the counts at pa1/pa4 is less than zero, take the background as the difference
    delta_counts_pa1 = np.max(np.stack([delta_counts_pa1, counts_pa1_bgnd_mean]).T, axis=1)
    delta_counts_pa4 = np.max(np.stack([delta_counts_pa4, counts_pa4_bgnd_mean]).T, axis=1)

    # If the difference between the counts at pa1/pa4 is zero then sustitute with the computer eps
    eps = float(np.finfo(np.float64).eps)
    delta_counts_pa1[delta_counts_pa1 == 0.] = eps
    delta_counts_pa4[delta_counts_pa4 == 0.] = eps

    delta_counts_pa1_err = np.linalg.norm(np.stack([counts_pa1_3300K_se, counts_pa1_bgnd_se]).T, axis=1)
    delta_counts_pa4_err = np.linalg.norm(np.stack([counts_pa4_3300K_se, counts_pa4_bgnd_se]).T, axis=1)

    # delta_counts_pa1_err[delta_counts_pa1_err == 0] = eps

    # print(delta_counts_pa1[delta_counts_pa1 <= 0.])


    calibration_df = pd.DataFrame(data={
        'Wavelength (nm)': wl_pa1_3300K_mean,
        'CPS @pregain 1': counts_pa1_3300K_mean,
        'CPS @pregain 4': counts_pa4_3300K_mean,
        'CPS @pregain 1 bgnd': counts_pa1_bgnd_mean,
        'CPS @pregain 4 bgnd': counts_pa4_bgnd_mean,
        'CPS @pregain 1 SE': counts_pa1_3300K_se,
        'CPS @pregain 4 SE': counts_pa4_3300K_se,
        'CPS @pregain 1 bgnd SE': counts_pa1_bgnd_se,
        'CPS @pregain 4 bgnd SE': counts_pa4_bgnd_se
    })


    # trans = transmission_dirty_window(wavelength=wl_pa1_3300K_mean)
    # # print(trans.shape)
    # calibration_df['Window transmission'] = trans
    radiance_at_wl = radiance_ls_interp(wl_pa1_3300K_mean) # wavelength is the same for all calibration spectra
    radiance_pag_1 = radiance_at_wl * np.power(
        delta_counts_pa1, -1.
    )

    radiance_pag_4 = radiance_at_wl * np.power(
        delta_counts_pa4, -1.
    )

    # Smooth the curve
    radiance_pag_1 = savgol_filter(
        radiance_pag_1,
        window_length=93,
        polyorder=3
    )

    radiance_pag_4 = savgol_filter(
        radiance_pag_4,
        window_length=93,
        polyorder=3
    )

    radiance_pag_1_err = radiance_pag_1 * np.abs(delta_counts_pa1_err / delta_counts_pa1)
    radiance_pag_4_err = radiance_pag_4 * np.abs(delta_counts_pa4_err / delta_counts_pa4)

    calibration_df['Radiance @pregain 1 (W/sr/cm^2/nm)'] = radiance_pag_1

    calibration_df['Radiance @pregain 1 error (W/sr/cm^2/nm)'] = radiance_pag_1_err

    calibration_df['Radiance @pregain 4 (W/sr/cm^2/nm)'] = radiance_pag_4

    calibration_df['Radiance @pregain 4 error (W/sr/cm^2/nm)'] = radiance_pag_4_err

    calibration_df.to_csv(r'./data/echelle_calibration_20240910.csv', index=False)


    for ax in axes:
        ax.legend(loc='upper left', ncols=1, fontsize=9)
        ax.xaxis.set_ticks_position('bottom')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(100.))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(50.))
        mf = ticker.ScalarFormatter(useMathText=True)
        mf.set_powerlimits((-2, 2))
        ax.yaxis.set_major_formatter(mf)
        ax.ticklabel_format(useMathText=True)
        ax.set_xlim(150, 1050)

    axes[0].yaxis.set_major_locator(ticker.MultipleLocator(2E5))
    axes[0].yaxis.set_minor_locator(ticker.MultipleLocator(1E5))
    axes[0].set_ylim(0, 1.05E6)


    axes[1].yaxis.set_major_locator(ticker.MultipleLocator(5E3))
    axes[1].yaxis.set_minor_locator(ticker.MultipleLocator(1E3))
    axes[1].set_ylim(0, 1.4E4)

    axes[1].set_xlabel(r"$\lambda$ {\sffamily (nm)}", usetex=True)
    # axes[1].set_xlabel(r"$\lambda$ (nm)}")

    axes[0].set_title('Lab sphere 3300 K')
    axes[1].set_title('Background (dark)')

    fig_raw.supylabel('Counts/s')

    fig_raw.savefig(r'./figures/labsphere_echelle_calibration_cps_20240910.png', dpi=600)

    fig_mean, axes = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
    fig_mean.set_size_inches(5.5, 5.5)

    axes[0].plot(wl_pa1_3300K_mean, counts_pa1_3300K_mean, color='C0', label='Preamp gain: 1')
    axes[0].fill_between(
        wl_pa1_3300K_mean, counts_pa1_3300K_mean - counts_pa1_3300K_se, counts_pa1_3300K_mean + counts_pa1_3300K_se,
        color='C0', alpha=0.25
   )

    axes[0].plot(wl_pa1_3300K_mean, counts_pa4_3300K_mean, color='C1', label='Preamp gain: 4')
    axes[0].fill_between(
        wl_pa1_3300K_mean, counts_pa4_3300K_mean - counts_pa4_3300K_se, counts_pa4_3300K_mean + counts_pa4_3300K_se,
        color='C1', alpha=0.25
    )

    axes[1].plot(wl_pa1_3300K_mean, counts_pa1_bgnd_mean, color='C2', label='Preamp gain: 1')
    axes[1].fill_between(
        wl_pa1_3300K_mean, counts_pa1_bgnd_mean - counts_pa1_bgnd_se, counts_pa1_bgnd_mean + counts_pa1_bgnd_se,
        color='C2', alpha=0.25
    )

    axes[1].plot(wl_pa1_3300K_mean, counts_pa4_bgnd_mean, color='C3', label='Preamp gain: 4')
    axes[1].fill_between(
        wl_pa1_3300K_mean, counts_pa4_bgnd_mean - counts_pa4_bgnd_se, counts_pa4_bgnd_mean + counts_pa4_bgnd_se,
        color='C3', alpha=0.25
    )

    axes[0].set_title('Lab sphere (3300 K)')
    axes[1].set_title('Background (dark)')

    for ax in axes:
        ax.legend(loc='upper left', ncols=1, fontsize=9)
        ax.xaxis.set_ticks_position('bottom')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(100.))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(50.))
        mf = ticker.ScalarFormatter(useMathText=True)
        mf.set_powerlimits((-2, 2))
        ax.yaxis.set_major_formatter(mf)
        ax.ticklabel_format(useMathText=True)
        ax.set_xlim(300, 1050)

    # axes[0].yaxis.set_major_locator(ticker.MultipleLocator(2E5))
    # axes[0].yaxis.set_minor_locator(ticker.MultipleLocator(1E5))
    # axes[0].set_ylim(0, 1.05E6)
    #
    # axes[1].yaxis.set_major_locator(ticker.MultipleLocator(5E3))
    # axes[1].yaxis.set_minor_locator(ticker.MultipleLocator(1E3))
    # axes[1].set_ylim(0, 1.4E4)

    axes[1].set_xlabel(r"$\lambda$ {\sffamily (nm)}", usetex=True)
    # axes[1].set_xlabel(r"$\lambda$ (nm)}")

    axes[0].set_title('Lab sphere 3300 K')
    axes[1].set_title('Background (dark)')
    fig_mean.supylabel('Counts/s')

    fig_mean.savefig(r'./figures/labsphere_echelle_calibration_cps_mean_20240910.png', dpi=600)

    fig_cal, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
    fig_cal.set_size_inches(5.5, 5.5)
    fig_cal.subplots_adjust(hspace=0, left=0.15, right=0.95, bottom=0.125, top=0.95)

    axes[0].plot(
        wl_pa1_3300K_mean, radiance_pag_1, color='C0', label='Preamp gain: 1'
    )

    axes[0].fill_between(
        wl_pa1_3300K_mean, radiance_pag_1 - radiance_pag_1_err, radiance_pag_1 + radiance_pag_1_err,
        color='C0', alpha=0.25
    )

    axes[1].plot(
        wl_pa1_3300K_mean, radiance_pag_4, color='C1', label='Preamp gain: 4'
    )

    axes[1].fill_between(
        wl_pa1_3300K_mean, radiance_pag_4 - radiance_pag_4_err, radiance_pag_4 + radiance_pag_4_err,
        color='C1', alpha=0.25
    )

    # axes[0].set_title('Preamp gain 1')
    # axes[1].set_title('Preamp gain 4')

    fig_cal.supylabel(r'{\sffamily (W/cm\textsuperscript{2}/nm/ster)/(Counts/s)}', usetex=True)
    axes[1].set_xlabel(r"$\lambda$ {\sffamily (nm)}", usetex=True)

    for ax in axes:
        ax.legend(loc='upper left', ncols=1, fontsize=9)
        # ax.xaxis.set_ticks_position('bottom')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(100.))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(50.))
        # mf = ticker.ScalarFormatter(useMathText=True)
        # mf.set_powerlimits((-2, 2))
        # ax.yaxis.set_major_formatter(mf)
        ax.ticklabel_format(useMathText=True)
        ax.set_yscale('log')
        ax.set_xlim(300, 1050)

    fig_cal.suptitle('Spectrometer calibration')
    fig_cal.savefig('./figures/labsphere_echelle_calibration_cps2radiance_20240910.png', dpi=600)

    plt.show()




if __name__ == '__main__':
    main()


