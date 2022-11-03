import h5py
import numpy as np
import matplotlib.pylab as plt
import os
from heat_flux_adi import simulate_adi_temp
import json
import matplotlib as mpl
import shutil
from matplotlib.animation import FFMpegWriter
import matplotlib.animation as manimation


base_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\heat_flux_calibration\results'
load_model = True
saved_h5 = 'ADI_k1_7.41E-01_chi_0.60_P4.70E+03'

qmax = 4.7E3

emissivity = 1.0 - (36.9 / 100)
reflectance = 40.4

M = 200  # number of intervals in r
N = 400  # the number of intervals in x
R = 1.27  # The radius of the cylinder in cm
R_sample = 0.5 * 1.288  #  # the radius of the sample in cm
L = 5.0  # the length of the cylinder in cm
holder_thickness = 1.27  # the radius of the stainless steel holder
dt = 1.0E-3  # s
beam_diameter = 1.5 * 0.8165  # cm
probe_size = 2.0  # mm

thermography_spot_diameter = 0.8  # cm
density_g = 1.698  # g/cm^3
""" 
Values of heat capacity for all types of natural and manufactured graphites are basically
the same, except near absolute-zero temperatures.

https://www.goodfellow.com/us/en-us/displayitemdetails/p/c-00-rd-000130/carbon-rod 

and 

https://poco.entegris.com/content/dam/poco/resources/reference-materials/brochures/brochure-graphite-properties-and-characteristics-11043.pdf
"""
# specific_heat_g = 0.712 # J / g / K
specific_heat_g = 0.6752  # Markelov, Volga, et al., 1973
k0_1 = 85E-2  # W / (cm K) https://www.graphitestore.com/core/media/media.nl?id=6310&c=4343521&h=Tz5uoWvr-nhJ13GL1b1lG8HrmYUqV1M_1bOTFQ2MMuiQapxt # GR001C
k0_2 = 16.2E-2  # W / (cm K)

kappa_1 = k0_1 / (density_g * specific_heat_g)
kappa_2 = 4.5E-2  # Thermal diffusivity of steel in cm^2/s
chi = 1.0 - (reflectance / 100.0)
T_a = 20.0  # ambient temperature in °C
pulse_length = 0.5  # in seconds
t_max = 2.01  # Maximum simulation time

x_tc_1 = 1.0  # position of temperature probe 1 in cm
x_tc_2 = 2.0  # position of temperature probe 2 in cm


# Kim Argonne National Lab 1965
def cp_ss304l(temperature):
    return 4.184 * (0.1122 + 3.222E-5 * temperature)


def rho_ss304l(temperature):
    return 7.9841 - 2.6506E-4 * temperature - 1.1580E-7 * temperature ** 2.0


def thermal_conductivity_ss304l(temperature):
    return 8.11E-2 + 1.618E-4 * temperature


k0_2 = thermal_conductivity_ss304l(T_a + 273.15)
cp_2 = cp_ss304l(T_a + 273.15)
rho_2 = rho_ss304l(T_a + 273.15)
kappa_2 = k0_2 / (cp_2 * rho_2)

if __name__ == "__main__":
    adi_data_dir = os.path.join(os.path.join(base_path, 'adi_data'))

    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    if not load_model:
        hf_file = simulate_adi_temp(
            laser_power=qmax, r_holder=R, r_sample=R_sample,
            length=L, kappa_1=kappa_1, kappa_2=kappa_2,
            k0_1=k0_1, k0_2=k0_2, r_points=M, x_points=N,
            pulse_length=pulse_length, dt=dt, chi=chi, T_a=T_a, t_max=t_max,
            report_every=20, debug=True, holder_thickness_cm=holder_thickness,
            save_h5=True, beam_diameter=beam_diameter, x_tc_1=x_tc_1, x_tc_2=x_tc_2,
            emissivity=1.0
        )

        if not os.path.exists(adi_data_dir):
            os.makedirs(adi_data_dir)
        shutil.move(hf_file + '.h5', os.path.join(adi_data_dir, hf_file + '.h5'))
    else:
        hf_file = saved_h5

    hf_file_path = os.path.join(adi_data_dir, saved_h5 + '.h5')
    print(f'Path to .h5 file: {hf_file_path}')

    with h5py.File(hf_file_path, 'r') as hf:
        d = hf.get('/data')
        M = d.attrs['M']  # number of intervals in r
        N = d.attrs['N']  # the number of intervals in x
        R = d.attrs['R']  # The radius of the sample holder in cm
        R_sample = hf['/data'].attrs['R_sample']  # The radius of the sample in cm
        x = np.array(d.get('x'))
        r = np.array(d.get('r'))
        L = x.max()  # the length of the sample
        dx = d.attrs['dx']
        dr = d.attrs['dr']
        dt = d.attrs['dt']
        idx_r = d.attrs['idx_r']
        elapsed_time = np.array(d.get('time'))
        x = np.array(hf['data/x'])
        r = np.array(hf['data/r'])

    probe_size_idx = int(probe_size * 0.1 / dx)
    probe_idx_delta = int(0.5 * probe_size_idx)

    idx_pd_spot = (np.abs(r - thermography_spot_diameter * 0.5)).argmin()
    tp1 = T_a * np.ones_like(elapsed_time)
    tp2 = T_a * np.ones_like(elapsed_time)
    xp1, xp2 = 1.0, x.max()
    idx_p1, idx_p2 = int(xp1 / dx), int(xp2 / dx)

    # The average temperature at the front surfacve
    t_front = T_a * np.ones_like(elapsed_time)

    for i in range(len(tp1)):
        ds_name = f'data/T_{i:d}'
        with h5py.File(hf_file_path, 'r') as hf:
            u = np.array(hf.get(ds_name))
            tp1[i] = u[idx_r, idx_p1]
            tp2[i] = u[idx_r, idx_p2]
            # t_front[i] = u[0:idx_r,0].mean()
            t_front[i] = u[0:5, 0].mean()

    fig1, ax1 = plt.subplots(ncols=1, constrained_layout=True)
    fig1.set_size_inches(4.5, 3.5)
    ax1.plot(elapsed_time, t_front, label=f'Mean front surface', c='tab:red')
    ax1.plot(elapsed_time, tp1, label=f'r={R_sample:.3f} cm, x={xp1:.2f} cm', c='tab:orange')
    ax1.plot(elapsed_time, tp2, label=f'r={R_sample:.3f} cm, x={xp2:.2f} cm', c='tab:green')

    leg = ax1.legend(
        loc='upper right', ncol=1
    )

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_xlim((0., t_max))


    def update_line(n, line, n_max):
        global cs, c, u_min, u_max
        u = None
        with h5py.File(hf_file_path, 'r') as hf:
            ds_name = f'data/T_{n:d}'
            u = np.array(hf.get(ds_name))
            t = hf[ds_name].attrs['time (s)']
        u_r = u[::-1, ].copy()
        u_plot = np.vstack((u_r[1::, :], u))

        line[0].set_array(u_plot[:, :].flatten())
        time_txt = f'{t:.3f} (s)'
        line[1].set_text(time_txt)
        if n % 10 == 0:
            print('Updating time step {0}/{1}'.format(n, n_max))
        return line


    n_max = int(t_max / dt)

    with h5py.File(hf_file_path, 'r') as hf:
        t_step = n_max - 1
        ds_name = f'data/T_{t_step:d}'
        u = np.array(hf.get(ds_name))
        t = hf[ds_name].attrs['time (s)']
        u_min = hf['data'].attrs['T_min']
        u_max = hf['data'].attrs['T_max']
        x = np.array(hf['data/x'])
        r = np.array(hf['data/r'])
        dr = float(hf['/data'].attrs['dr'])

    for i in range(n_max):
        with h5py.File(hf_file_path, 'r') as hf:
            ds_name = f'data/T_{i:d}'
            u = np.array(hf.get(ds_name))
            t = hf[ds_name].attrs['time (s)']
            u_min = hf['data'].attrs['T_min']
            u_max = hf['data'].attrs['T_max']

    fig, ax = plt.subplots(ncols=1, constrained_layout=True)
    r_plot = dr*np.arange(-M, M + 1)
    x_plot = x
    u_r = u[::-1, :].copy()
    print(f'T_min, TMax: {u_min:.2f}, {u_max:.2f}')

    # r_start_idx = 1 if len(r) % 2 == 0 else 0
    u_plot = np.vstack((u_r[1:, :], u))

    print(f'Shape of x: {x.shape}')
    print(f'Shape of r: {r.shape}')
    print(f'Shape of u_plot: {u_plot.shape}')
    cs = ax.pcolormesh(
        x_plot, r_plot, u_plot[:, :], cmap=plt.cm.jet, vmin=u_min, vmax=u_max, shading='gouraud', rasterized=True
    )
    cbar = fig.colorbar(cs)
    cbar.ax.set_ylabel('Temperature (°C)')
    cbar.ax.set_ylim(u_min, u_max)
    cbar.formatter.set_powerlimits((-3, 3))
    cbar.formatter.useMathText = True
    cbar.update_ticks()
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('r (cm)')

    time_txt = ax.text(
        0.95, 0.95, '0.000',
        horizontalalignment='right',
        verticalalignment='top',
        color='w',
        transform=ax.transAxes
    )

    line = [cs, time_txt]

    metadata = dict(title='Heat Diffusion Movie', artist='Matplotlib',
                    comment=f'kappa_1 = {kappa_1:.3E}, kappa_2 = {kappa_2:.3E}')
    plt.rcParams['animation.ffmpeg_path'] = r'C:\FFmpeg\bin\ffmpeg.exe'
    writer = FFMpegWriter(fps=24, metadata=metadata)

    n_max = int(t_max / dt)
    ani = manimation.FuncAnimation(
        fig, update_line, interval=300,
        repeat=False, frames=np.arange(0, n_max, 1),
        fargs=(line, n_max)
    )

    ft = os.path.splitext(hf_file_path)[0] + '_movie.mp4'
    print(f'Movie file: {ft}')
    ani.save(ft, writer=writer, dpi=200)

    plt.show()
