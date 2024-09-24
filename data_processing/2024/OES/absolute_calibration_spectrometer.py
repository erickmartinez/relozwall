import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import json
from scipy.interpolate import interp1d
import data_processing.secrets as my_secrets
import data_processing.echelle as ech
import os
from scipy.stats.distributions import t

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
    # photon_energy_ev = 1239.84198433 * np.power(wl_ls, -1.)
    # photon_energy_J = 19.86445857E-17 * np.power(wl_ls, -1.)
    photon_flux_ls = 5.03411656E15 * wl_ls * radiance_ls
    # print(np.isclose(photon_flux_ls, radiance_ls / photon_energy_J, atol=1E-5))
    flux_ls_interp = interp1d(x=wl_ls, y=photon_flux_ls)

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
        counts_ps = df['counts'].values / exposure_s / accumulations
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
    wl_pa4_3300K_mean = wl_pa4_3300K.mean(axis=1)
    wl_pa1_bgnd_mean = wl_pa1_bgnd.mean(axis=1)
    wl_pa4_bgnd_mean = wl_pa4_bgnd.mean(axis=1)

    counts_pa1_3300K_mean = counts_pa1_3300K.mean(axis=1)
    counts_pa4_3300K_mean = counts_pa4_3300K.mean(axis=1)
    counts_pa1_bgnd_mean = counts_pa1_bgnd.mean(axis=1)
    counts_pa4_bgnd_mean = counts_pa4_bgnd.mean(axis=1)

    counts_pa1_3300K_std = counts_pa1_3300K.std(ddof=1, axis=1)
    counts_pa4_3300K_std = counts_pa4_3300K.std(ddof=1, axis=1)
    counts_pa1_bgnd_std = counts_pa1_bgnd.std(ddof=1, axis=1)
    counts_pa4_bgnd_std = counts_pa4_bgnd.std(ddof=1, axis=1)

    confidence_level = 0.95
    alpha = 1. - confidence_level
    tval = t.ppf(1 - 0.5 * alpha, 2)
    se_factor = tval / np.sqrt(3)

    counts_pa1_3300K_se = counts_pa1_3300K_std * se_factor
    counts_pa4_3300K_se = counts_pa4_3300K_std * se_factor
    counts_pa1_bgnd_se = counts_pa1_bgnd_std * se_factor
    counts_pa4_bgnd_se = counts_pa4_bgnd_std * se_factor

    # print('wl1 3300K equal to wl4 3300K?', np.isclose(wl_pa1_3300K_mean, wl_pa4_3300K_mean).all())
    # print('wl1 3300K equal to wl1 bgnd?', np.isclose(wl_pa1_3300K_mean, wl_pa1_bgnd_mean).all())
    # print('wl1 3300K equal to wl2 bgnd?', np.isclose(wl_pa1_3300K_mean, wl_pa4_bgnd_mean).all())


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

    trans = transmission_dirty_window(wavelength=wl_pa1_3300K_mean)
    # # print(trans.shape)
    calibration_df['Window transmission'] = trans
    flux_at_wl = flux_ls_interp(wl_pa1_3300K_mean)
    flux_pag_1 = trans * flux_at_wl * np.power(
        counts_pa1_3300K_mean - counts_pa1_bgnd_mean, -1.
    )

    flux_pag_4 = trans * flux_at_wl * np.power(
        counts_pa4_3300K_mean - counts_pa4_bgnd_mean, -1.
    )
    calibration_df['Flux @pregain 1 (Photons/s/sr/cm^2/nm)'] = flux_pag_1

    calibration_df['Flux @pregain 1 error (Photons/s/sr/cm^2/nm)'] = flux_pag_1 * np.linalg.norm(
        np.stack([counts_pa1_3300K_se, counts_pa1_bgnd_se]).T, axis=1
    )

    calibration_df['Flux @pregain 4 (Photons/s/sr/cm^2/nm)'] = flux_pag_4

    calibration_df['Flux @pregain 4 error (Photons/s/sr/cm^2/nm)'] = flux_pag_4 * np.linalg.norm(
        np.stack([counts_pa4_3300K_se, counts_pa4_bgnd_se]).T, axis=1
    )


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

    fig_mean.savefig(r'./figures/labsphere_echelle_calibration_cps_mean_20240910.png', dpi=600)

    fig_cal, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
    fig_cal.set_size_inches(5.5, 5.5)
    fig_cal.subplots_adjust(hspace=0, left=0.15, right=0.95, bottom=0.125, top=0.95)

    axes[0].plot(
        wl_pa1_3300K_mean, flux_pag_1, color='C0', label='Preamp gain: 1'
    )

    # axes[0].fill_between(
    #     wl_pa1_3300K_mean, flux_pag_1 - calibration_df['Flux @pregain 1 error (Photons/s/sr/cm^2/nm)'].values,
    #     wl_pa1_3300K_mean, flux_pag_1 + calibration_df['Flux @pregain 1 error (Photons/s/sr/cm^2/nm)'].values,
    #     color='C0', alpha=0.25
    # )

    axes[1].plot(
        wl_pa1_3300K_mean, flux_pag_4, color='C1', label='Preamp gain: 1'
    )

    # axes[1].fill_between(
    #     wl_pa1_3300K_mean, flux_pag_4 - calibration_df['Flux @pregain 4 error (Photons/s/sr/cm^2/nm)'].values,
    #     wl_pa1_3300K_mean, flux_pag_4 + calibration_df['Flux @pregain 4 error (Photons/s/sr/cm^2/nm)'].values,
    #     color='C1', alpha=0.25
    # )

    # axes[0].set_title('Preamp gain 1')
    # axes[1].set_title('Preamp gain 4')

    fig_cal.supylabel('(Photons/s/cm^2/nm/ster)/(Counts/s)')
    axes[1].set_xlabel(r"$\lambda$ {\sffamily (nm)}", usetex=True)

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

    plt.show()




if __name__ == '__main__':
    main()


