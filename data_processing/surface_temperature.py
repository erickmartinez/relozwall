import h5py
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import os
from matplotlib import ticker, gridspec
import matplotlib as mpl
from utils import get_experiment_params, correct_thermocouple_response, lighten_color
import json
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from heat_flux_adi import gaussian_beam

# graphite_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\surface_temperature\equilibrium'
# sample_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\IR_VS_POWER'
graphite_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\surface_temperature\equilibrium_redone'
sample_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\surface_temperature\equilibrium_redone\pebble_sample'
graphite_filetag = 'graphite_equilibrium_redone_files'
sample_filetag = 'pebble_sample_equilibrium_redone_files'
pebble_velocity_csv = 'velocity_database.csv'
filetag = 'surface_temperature_plot'
graphite_heat_analysis_csv = 'graphite_heat_analysis.csv'
pebble_sample_heat_analysis_csv = 'pebble_sample_heat_analysis.csv'


heat_data_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\heat_flux_calibration\IR Thermography Calibration'
heat_data_file = r'LT_GR008G_6mTorr-contact-shield_100PCT_50GAIN 2022-05-04_1'
saved_h5 = 'ADI_k1_7.41E-01_chi_0.60_P4.70E+03'
p0 = 4.7E3
time_constant = 1.68
sample_diameter = 0.918 # 2.54*3/8

beam_diameter = 0.8164 * 1.5  #cm

if __name__ == '__main__':
    experiment_params = get_experiment_params(relative_path=heat_data_path, filename=heat_data_file)
    photodiode_gain = experiment_params['Photodiode Gain']['value']
    laser_power_setting = experiment_params['Laser Power Setpoint']['value']
    sample_name = experiment_params['Sample Name']['value']

    temperature_path = os.path.join(heat_data_path, 'temperature_data', f'{sample_name.upper()}_{laser_power_setting}')
    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    graphite_temperature_df = pd.read_csv(
        os.path.join(graphite_path, graphite_filetag + '_surface_temperature.csv')).apply(pd.to_numeric)

    sample_temperature_df = pd.read_csv(os.path.join(sample_path, sample_filetag + '_surface_temperature.csv')).apply(
        pd.to_numeric)
    sample_temperature_agg_df = sample_temperature_df.groupby('Laser power setpoint (%)').agg({
        'Heat flux (MW/m^2)': ['mean'], 'Max surface temperature (C)': ['mean', 'min']
    })
    pebble_velocity_df = pd.read_csv(os.path.join(sample_path, pebble_velocity_csv))
    pebble_velocity_df.iloc[:, 1:] = pebble_velocity_df.iloc[:, 1:].apply(pd.to_numeric)
    pebble_velocity_agg_df = pebble_velocity_df.groupby('Laser power setpoint (%)').agg({
        'Particle velocity mode (cm/s)': ['mean', 'min'], 'Particle velocity std (cm/s)': [
            'mean']
    })
    sample_outgassing_df = pd.read_csv(os.path.join(sample_path, sample_filetag + '_OUTGASSING.csv')).apply(
        pd.to_numeric)
    graphite_outgassing_df = pd.read_csv(os.path.join(graphite_path, graphite_filetag + '_OUTGASSING.csv')).apply(
        pd.to_numeric)

    laser_power_g = graphite_temperature_df['Laser power setpoint (%)'].values
    heat_flux_g = graphite_temperature_df['Heat flux (MW/m^2)'].values
    laser_power_s = sample_temperature_df['Laser power setpoint (%)'].values
    heat_flux_s = sample_temperature_agg_df['Heat flux (MW/m^2)']['mean'].values
    graphite_surface_temperature = graphite_temperature_df['Max surface temperature (C)'].values
    sample_surface_temperature = sample_temperature_agg_df['Max surface temperature (C)']['mean'].values

    pebble_velocity = pebble_velocity_agg_df['Particle velocity mode (cm/s)']['mean'].values
    pebble_velocity_std = pebble_velocity_agg_df['Particle velocity std (cm/s)']['mean'].values

    surface_temperature_df = pd.read_csv(
        os.path.join(temperature_path, f'{heat_data_file}_surface_temp.csv'), comment='#'
    ).apply(pd.to_numeric)

    measurement_time = surface_temperature_df['Time (s)'].values
    surface_temperature = surface_temperature_df['Surface Temperature (°C)'].values

    adi_data_dir = os.path.join(os.path.join(os.path.dirname(heat_data_path), 'results', 'adi_data'))
    hf_file_path = os.path.join(adi_data_dir, saved_h5 + '.h5')
    """
    These parameters musth match with what the ones used for the h5 simulation
    """
    with h5py.File(hf_file_path, 'r') as hf:
        d = hf.get('/data')
        M = d.attrs['M']  # number of intervals in r
        N = d.attrs['N']  # the number of intervals in x
        R = d.attrs['R']  # The radius of the sample holder in cm
        R_sample = hf['/data'].attrs['R_sample'] # The radius of the sample in cm
        x = np.array(d.get('x'))
        r = np.array(d.get('r'))
        L = x.max() # the length of the sample
        dx = d.attrs['dx']
        dr = d.attrs['dr']
        dt = d.attrs['dt']
        idx_r = d.attrs['idx_r']
        elapsed_time = np.array(d.get('time'))

    probe_size = 2.0  # mm
    t_max = 2.01
    T_a = 20.0
    thermography_spot_diameter = 0.8  # cm

    probe_size_idx = int(probe_size * 0.1 / dx)
    probe_idx_delta = int(0.5 * probe_size_idx)

    # x = dx * np.arange(0, N + 1)
    # r = dr * np.arange(0, M + 1)
    idx_pd_spot = (np.abs(r - thermography_spot_diameter * 0.5)).argmin()
    # Get the size of the time array
    # elapsed_time = np.arange(0, t_max + dt, dt, dtype=np.float64)
    # The temperature at the surface of the rod closest to the light source
    tp1 = T_a * np.ones_like(elapsed_time)
    xp1 = 1.0
    idx_p1 = int(xp1 / dx)

    # The average temperature at the front surfacve
    t_front = T_a * np.ones_like(elapsed_time)


    for i in range(len(tp1)):
        ds_name = f'data/T_{i:d}'
        with h5py.File(hf_file_path, 'r') as hf:
            u = np.array(hf.get(ds_name))
            tp1[i] = u[idx_r, idx_p1 - probe_idx_delta:idx_p1 + probe_idx_delta].mean()
            t_front[i] = u[0:idx_pd_spot, 0:3].mean()

    estimated_power_density = p0 * (1.0 - np.exp(-2.0 * (2.0 * R_sample / beam_diameter) ** 2.0)) / (
            np.pi * R_sample ** 2.0)

    tc_csv = os.path.join(heat_data_path, heat_data_file + '_tcdata.csv')
    tc_df = pd.read_csv(tc_csv, comment='#').apply(pd.to_numeric)
    tc_time = tc_df['Time (s)'].values
    temperature_a = tc_df['TC1 (C)'].values
    ta_corrected = correct_thermocouple_response(
        measured_time=tc_time, measured_temperature=temperature_a, tau=time_constant
    )

    tol = 0.25
    tc_0 = temperature_a[0:5].mean()
    print(f'TC[t=0]: {tc_0:4.2f} °C')
    msk_onset = (temperature_a - tc_0) > tol
    time_onset = tc_time[msk_onset]
    time_onset = time_onset[0]
    idx_onset = (np.abs(tc_time - time_onset)).argmin() - 20
    # print(idx_onset)

    tc_time = tc_time[idx_onset::]
    tc_time -= tc_time.min()
    temperature_a = temperature_a[idx_onset::]
    ta_corrected = ta_corrected[idx_onset::]
    tc_time_positive_idx = tc_time > 0
    tc_time = tc_time[tc_time_positive_idx]
    temperature_a = temperature_a[tc_time_positive_idx]
    ta_corrected = ta_corrected[tc_time_positive_idx]

    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(4.0, 5.0)

    gs = gridspec.GridSpec(ncols=1, nrows=2, figure=fig)  # , hspace=0.15)  # , height_ratios=[1.618, 1.618, 1])

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax1.plot(
        measurement_time, surface_temperature, ls='-', lw=4, label=f' IR Thermography (z = 0)',
        c='tab:purple', marker=None, fillstyle='none', ms=8
    )

    ax1.plot(tc_time, ta_corrected, label=f' Thermocouple (z = 1.0 cm)', ls='-', lw=4,
            color='tab:olive', marker=None, fillstyle='none', ms=8)

    ax1.plot(elapsed_time, t_front,
             # label=f'Fit',
             c=lighten_color('tab:purple', 1.4),
             ls='--', lw=1.5, label=' 2D heat diffusion model')
    ax1.plot(elapsed_time, tp1, #label='Fit',
             c=lighten_color('tab:olive', 1.4),
             ls='--', lw=1.5)

    ax1.set_title(
        f'Average power density: {estimated_power_density * 0.01:.0f} MW/m$^{{\\mathregular{{2}}}}$',
        fontweight='regular', #pad=20
    )

    prop = {'size': 9}

    leg = ax1.legend(
        loc='upper right', ncol=1, frameon=True,
        # bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
        # ncol=2, mode="expand", borderaxespad=0.,
        prop=prop
    )

    ax1.tick_params(axis='y', right=False, zorder=10, which='both')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_xlim((0., 2.0))
    ax1.set_ylim(bottom=0.0, top=2200)
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(0.125))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(400))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(200))

    ax2.plot(
        heat_flux_s, sample_surface_temperature, color='tab:red', label='Sample', marker='D', lw=1.5,
        # mec='k',
        fillstyle='none', mew=1.5
    )

    n_graphite = len(graphite_surface_temperature)
    n_sample = len(sample_surface_temperature)
    print(f'Sample data points: {n_sample}, Graphite data points: {n_graphite}')
    ax2.plot(
        heat_flux_g, graphite_surface_temperature, color='navy', label='Graphite', marker='^', lw=1.5,
        # mec='k',
        fillstyle='none', mew=1.5
    )

    ax2.set_xlabel('Heat load (MW/m$^{\\mathregular{2}}$)')
    ax2.set_ylabel('Temperature °C')
    ax2.set_title('Surface temperature')
    ax2.set_xlim(0, 50)
    ax2.set_ylim(bottom=1250, top=3000)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax2.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(500))
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(250))

    ax2.legend(loc='best', frameon=True, prop=prop)

    # plot_r = 1.8
    # n_points = 501
    # x = np.linspace(-plot_r, plot_r, num=n_points)
    # y = np.linspace(-plot_r, plot_r, num=n_points)
    # X, Y = np.meshgrid(x, y)
    # dx = x[1] - x[0]

    def get_r(x, y):
        return np.sqrt(x ** 2 + y ** 2)

    def get_q(x, y):
        return gaussian_beam(r=get_r(x, y), beam_diameter=beam_diameter, beam_power=p0)


    # q = get_q(X, Y)
    # q_min = q.flatten().min()
    # q_max = q.flatten().max()
    # def integrate_power(r:float):
    #     # xx, yy = X[0:n_points_1, x>0], Y[y>0, 0:n_points_1]
    #     # q1 = get_q(xx, yy)
    #     s = 0
    #     xx = x[x>=0]
    #     yy = y[y>=0]
    #     XX, YY = np.meshgrid(xx, yy)
    #     qq = get_q(XX, YY)
    #     r2 = r*r
    #     area = np.pi*r2
    #     for ii, xi in enumerate(xx):
    #         for jj, yj in enumerate(yy):
    #             if xi**2.0 + yj**2.0 <= r2:
    #                 # print(f'({ii}, {jj}), ({xi:.3f}, {yj:.3f}), X({ii},{jj}) = {XX[ii,jj]}, Y({ii},{jj}) ={YY[ii,jj]}, r: {np.sqrt(xi**2.0 + yj**2.0)}, q: {q[ii,jj]:.3E}')
    #                 s += qq[ii, jj]
    #     return 4.0*s*dx*dx/area
    #
    # print(f'Power for d = {2.0*R_sample:.2f} cm: {integrate_power(R_sample)*0.1:.2E} MW/m^2')
    # print(f'Power for d = {sample_diameter:.2f} cm: {integrate_power(0.5*sample_diameter) * 0.1:.2E} MW/m^2')


    # Create inset of width 1.3 inches and height 0.9 inches
    # at the default upper right location
    # axins1 = inset_axes(
    #     ax1, width=0.7, height=0.7, loc=1, borderpad=0,
    #     bbox_to_anchor=(0.98, 0.7),
    #     bbox_transform=ax1.transAxes
    # )
    # axins2 = inset_axes(ax2, width=0.7, height=0.7, loc=4, borderpad=0.5)
    # axins1.tick_params(labelleft=False, labelbottom=False, color='w', width=1.0, left=False, bottom=False)
    # axins2.tick_params(labelleft=False, labelbottom=False, color='w', width=1.0, left=False, bottom=False)
    # # axins.tick_params(axis='both', labelsize=8)
    # cs1 = axins1.pcolormesh(
    #     x, y, q, cmap=plt.cm.jet, vmin=q_min, vmax=q_max,
    #     shading='gouraud', rasterized=True
    # )
    # axins1.set_aspect('equal')
    # # axins1.set_xlabel('x (cm)', fontsize=6)
    # # axins1.set_ylabel('y (cm)', fontsize=6)
    # axins1.set_xlim(-1.0, 1.0)
    # axins1.set_ylim(-1.0, 1.0)
    #
    # axins1.set_title(f'Ø = {2.0*R_sample:.2f} cm', fontsize=8, fontweight='regular', pad=4)
    # cs2 = axins2.pcolormesh(
    #     x, y, q, cmap=plt.cm.jet, vmin=q_min, vmax=q_max,
    #     shading='gouraud', rasterized=True
    # )
    # axins2.set_aspect('equal')
    # # axins2.set_xlabel('x (cm)', fontsize=6)
    # # axins2.set_ylabel('y (cm)', fontsize=6)
    # axins2.set_xlim(-1.0, 1.0)
    # axins2.set_ylim(-1.0, 1.0)
    # axins2.set_title(f'Ø = {sample_diameter:.2f} cm', fontsize=8, fontweight='regular', pad=4)
    #
    # circle1 = plt.Circle((0, 0), 0.5 * 1.27, ec='r', fill=False, clip_on=False, ls=(0,(3,1)))
    # circle2 = plt.Circle((0, 0), 0.5*0.9, ec='r', fill=False, clip_on=False, ls=(0,(3,1)))
    # axins1.add_patch(circle1)
    # axins2.add_patch(circle2)

    # Add panel labels out of the box
    ax1.text(
        -0.1, 1.15, '(a)', transform=ax1.transAxes, fontsize=14, fontweight='bold',
        va='top', ha='right'
    )
    ax2.text(
        -0.1, 1.15, '(b)', transform=ax2.transAxes, fontsize=14, fontweight='bold',
        va='top', ha='right'
    )

    fig.savefig(os.path.join(sample_path, filetag + '.svg'), dpi=600)
    fig.savefig(os.path.join(sample_path, filetag + '.eps'), dpi=600)
    fig.savefig(os.path.join(sample_path, filetag + '.png'), dpi=600)

    fig2, axf2 = plt.subplots(nrows=2)
    fig2.set_size_inches(4.0, 4.5)

    axf2[0].plot(
        heat_flux_s, sample_surface_temperature, color='tab:red', label='Sample', marker='D', lw=1.5,
        # mec='k',
        fillstyle='none', mew=1.5
    )

    axf2[0].plot(
        heat_flux_g, graphite_surface_temperature, color='navy', label='Graphite', marker='^', lw=1.5,
        # mec='k',
        fillstyle='none', mew=1.5
    )

    axf2[1].set_xlabel('Heat flux (MW/m$^{\\mathregular{2}}$)')
    axf2[0].set_ylabel('Temperature °C')
    axf2[0].set_title('Surface temperature')
    axf2[0].set_xlim(0, 50)
    axf2[0].set_ylim(bottom=1250, top=3000)
    axf2[0].xaxis.set_major_locator(ticker.MultipleLocator(10))
    axf2[0].xaxis.set_minor_locator(ticker.MultipleLocator(5))
    axf2[0].yaxis.set_major_locator(ticker.MultipleLocator(500))
    axf2[0].yaxis.set_minor_locator(ticker.MultipleLocator(250))

    axf2[0].legend(loc='best', frameon=True, prop=prop)
    axf2[1].errorbar(
        heat_flux_s, pebble_velocity, yerr=pebble_velocity_std, c='lightblue', marker='^', lw=1.5,
        # mec='k',
        mew=1.25, capsize=3.5, capthick=1.25, ecolor='lightblue', fillstyle='none'
    )
    axf2[1].set_ylabel('cm/s')
    axf2[1].set_title('Pebble velocity ')
    axf2[1].set_ylim(bottom=0, top=75.0)
    axf2[1].yaxis.set_major_locator(ticker.MultipleLocator(25))
    axf2[1].yaxis.set_minor_locator(ticker.MultipleLocator(12.5))
    axf2[1].set_xlim(0, 50)
    axf2[1].xaxis.set_major_locator(ticker.MultipleLocator(10))
    axf2[1].xaxis.set_minor_locator(ticker.MultipleLocator(5))

    fig2.tight_layout()

    fig2.savefig(os.path.join(sample_path, filetag + '_pebble_velocity.pdf'), dpi=600)
    print('fig2 save path:')
    print(os.path.join(sample_path, filetag + '_pebble_velocity.pdf'))

    plt.show()
