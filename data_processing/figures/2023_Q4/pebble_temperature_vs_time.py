import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import os
import platform
from skimage.util import crop

from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats.distributions import t
from data_processing.utils import get_experiment_params, latex_float
import json
import cv2

platform_system = platform.system()
if platform_system != 'Windows':
    drive_path = r'/Users/erickmartinez/Library/CloudStorage/OneDrive-Personal'
else:
    drive_path = r'C:\Users\erick\OneDrive'

data_path = r'Documents/ucsd/Postdoc/research/manuscripts/paper2/figure_prep/pebble temperature'
tracking_csv = r'Documents/ucsd/Postdoc/research/manuscripts/paper2/figure_prep/Firing Tests - Mass Loss - Tracking_gsr_20231128_mod.csv'
laser_power_dir = r'Documents\ucsd\Postdoc\research\data\firing_tests\MATERIAL_SCAN\laser_output'
deposition_rates_csv = r'Documents/ucsd/Postdoc/research/manuscripts/paper2/figure_prep/deposition_rates_20231128.csv'
firing_csv = r'Documents/ucsd/Postdoc/research/data/firing_tests/merged_db.xlsx'
path_to_tracking_data = r'Documents/ucsd/Postdoc/research/data/firing_tests/SS_TUBE/GC'
calibration_path = r'Documents\ucsd\Postdoc\research\thermal camera\calibration\CALIBRATION_20231010'

beam_radius = 0.5 * 0.8165  # * 1.5 # 0.707
sample_diameter = 1.025

row_id_s = 426
row_id_b = 435

"""
Parameters for the deposition cone
"""
h0 = 26.67  # cm
nn = 7.4
r_viewport = 12.5

"""
Absorption coefficient to correct signal
"""
absorption_coefficient = 5.740E-3  # /nm
"""
Parameters to estimate the graphite sublimation rates
E_a: 8.2321 eV
r_0: 2.7424E+18 C/s/nm^2
"""
e_a = 8.231
r_0_c = 2.7424E+18
torrL2atoms = 3.219E19
r_0 = 2.7424 / 32.19 * 1E18  # (Torr-L/s/m^2)

crop_image = True
center = np.array([280, 484])
crop_r = 200  # 9x
crop_extents = {
    'left': (center[0] - crop_r),
    'top': center[1] - crop_r,
    'right': 1440 - (center[0] + crop_r),
    'bottom': 1080 - (center[1] + crop_r)
}

pixel_size = 20.4215  # pixels/mm
px2mm = 1. / pixel_size
px2cm = 0.1 * px2mm


def gaussian_beam_aperture_factor(beam_radius, sample_radius):
    return 1.0 - np.exp(-2.0 * (sample_radius / beam_radius) ** 2.0)


def pd_norm(x):
    return np.linalg.norm(x)


map_sample_suffix = {
    1: 15, 2: 30, 3: 60, 4: 120, 5: 180
}


def sample_id_suffix_to_baking_temp(ss):
    global map_sample_suffix
    r = 15 * np.ones(len(ss))
    for i, si in enumerate(ss):
        if not pd.isna(si):
            r[i] = map_sample_suffix[int(pd.to_numeric(si))]
    return r


def find_file_that_starts_with(relative_path, starts_with):
    file_list = [f for f in os.listdir(relative_path) if f.startswith(starts_with)]
    return file_list[0]


def map_laser_power_settings():
    global laser_power_dir, drive_path, beam_radius
    if platform.system() != 'Windows':
        laser_power_dir = laser_power_dir.replace('\\', '/')
    rdir = os.path.join(drive_path, laser_power_dir)

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


def normalize_path(the_path):
    global platform_system, drive_path
    the_path = os.path.join(drive_path, the_path)
    if platform_system != 'Windows':
        the_path = the_path.replace('\\', '/')
    return the_path


def load_calibration(calibration_csv):
    df = pd.read_csv(calibration_csv).apply(pd.to_numeric)
    return df['Temperature [K]'].values


def modified_knudsen(r_, h0_=h0, n_=nn):
    cos_q = h0_ / np.sqrt(r_ ** 2. + h0_ ** 2.)
    return np.power(cos_q, n_ + 2.)


cone_factor = modified_knudsen(r_=r_viewport)


def correct_for_window_deposit_intensity(img: np.ndarray, t, rate_d):
    global absorption_coefficient, r_viewport, cone_factor
    return min(np.power(10., absorption_coefficient * t * rate_d * cone_factor) * img, 255)


def convert_to_temperature(signal, cali: np.ndarray, dt, emission_time, rate_d) -> np.ndarray:
    n, m = signal.shape
    temp_img = np.zeros((n, m), dtype=float)
    for i in range(n):
        for j in range(m):
            s = correct_for_window_deposit_intensity(signal[i, j], dt, rate_d)
            if dt <= emission_time + 0.005:
                s = correct_for_window_deposit_intensity(signal[i, j], emission_time, rate_d)
            temp_img[i, j] = cali[int(s)]
    return temp_img  # .astype(np.uint8)


def get_sublimation_rate(temperature_k, r0=r_0, ea=e_a):
    return r0 * np.exp(-ea / (8.617333262e-05 * temperature_k))


def sublimation_rate_thermal_img(t_img, threshold=0):
    n, m = t_img.shape
    g_rate = 0.
    cnt = 0
    hot_area = 0
    for i in range(n):
        for j in range(m):
            temp_i = t_img[i, j]
            if temp_i > threshold:
                cnt += 1
                sr = get_sublimation_rate(t_img[i, j])
                hot_area += px2cm ** 2.
                g_rate += sr
    return g_rate / cnt,  hot_area


def get_cropped_image(img) -> np.ndarray:
    width, height = img.shape
    left = int(crop_extents['left'])  # * width)
    top = int(crop_extents['top'])  # * height)
    right = int(crop_extents['right'])  # * width)
    bottom = int(crop_extents['bottom'])  # * height)
    img2 = crop(img, ((top, bottom), (left, right)), copy=True)
    return img2


def threshold_image(signal, threshold=0):
    msk = get_img_msk(signal, threshold)
    n, m = signal.shape
    temp_img = np.zeros((n, m), dtype=float)
    for i in range(n):
        for j in range(m):
            if msk[i, j]:
                temp_img[i, j] = signal[i, j]
    return temp_img


def get_img_msk(signal, threshold=0) -> np.ndarray:
    return signal > threshold  # .astype(np.uint8)


def main():
    global data_path, deposition_rates_csv, laser_power_dir, tracking_csv, firing_csv, path_to_tracking_data
    global calibration_path
    base_path = normalize_path(base_path)
    deposition_rates_csv = normalize_path(deposition_rates_csv)
    laser_power_dir = normalize_path(laser_power_dir)
    tracking_csv = normalize_path(tracking_csv)
    firing_csv = normalize_path(firing_csv)
    path_to_tracking_data = normalize_path(path_to_tracking_data)
    calibration_path = normalize_path(calibration_path)

    # load transmission measurements
    transmission_df = pd.read_csv(deposition_rates_csv)
    transmission_df = transmission_df.loc[:, ~transmission_df.columns.str.contains('Unnamed')]
    transmission_columns = transmission_df.columns
    transmission_df[transmission_columns[2::]] = transmission_df[transmission_columns[2::]].apply(pd.to_numeric)

    transmission_df[['Small spheres', 'Big spheres']] = transmission_df['Pebble material'].str.extract(
        r'\[(\d+)\,(\d+)\]').apply(pd.to_numeric)

    transmission_df[['Baking time (min)']] = transmission_df['Sample ID'].str.extract(
        r'R\d+N\d+\-?(\d+)?'
    ).apply(sample_id_suffix_to_baking_temp)

    transmission_df['Small spheres'].fillna(0, inplace=True)
    transmission_df['Big spheres'].fillna(100, inplace=True)

    transmission_df.sort_values(by=['Big spheres', 'Laser power setting (%)'], inplace=True)
    print(transmission_df)

    laser_power_mapping = map_laser_power_settings()
    lp = np.array([laser_power_mapping[int(x)] for x in transmission_df['Laser power setting (%)']])
    sa = 0.25 * np.pi * np.power(sample_diameter, 2.)
    transmission_df['Sample area (cm^2)'] = sa
    transmission_df['Laser power [MW/m2]'] = lp * gaussian_beam_aperture_factor(beam_radius,
                                                                                0.5 * sample_diameter) / sa / 100.
    transmission_df['Graphite sublimation (Torr-L/s/m^2)'] = np.nan

    tracking_df = pd.read_csv(tracking_csv)
    # print(f'len(tracking_df): {len(tracking_df)}')
    tracking_df_cols = tracking_df.columns
    tracking_df[tracking_df_cols[1:5]] = tracking_df[tracking_df_cols[1:5]].apply(pd.to_numeric)
    tracking_df[tracking_df_cols[6::]] = tracking_df[tracking_df_cols[6::]].apply(pd.to_numeric)

    tracking_df[['Sample ID', 'ROW', 'Laser power setting (%)']] = tracking_df['Test'].str.extract(
        r'LCT\_(R\d+N\d+)\-?\d*\_ROW(\d+)\_(\d+)PCT\_.*$'
    )

    tracking_df[['ROW', 'Laser power setting (%)']] = tracking_df[['ROW', 'Laser power setting (%)']].apply(
        pd.to_numeric)
    tracking_df.reset_index(inplace=True, drop=True)
    # print(f'len(tracking_df): {len(tracking_df)}')

    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)
    norm1 = plt.Normalize(vmin=1500, vmax=3200)

    sample_area_cm2 = 0.25 * np.pi * sample_diameter ** 2.

    fig_bs, ax_bs = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    # fig_bs.subplots_adjust(hspace=0)
    fig_bs.set_size_inches(4., 3.)
    ax_bs.set_title(r'35 MW/m$^{\mathregular{2}}$')
    ax_bs.set_xlabel('t [s]')
    ax_bs.set_ylabel('T [K]')
    ax_bs.set_ylim(1000, 3000)
    ax_bs.yaxis.set_major_locator(ticker.MultipleLocator(500))
    ax_bs.yaxis.set_minor_locator(ticker.MultipleLocator(100))
    ax_bs.set_xlim(0, 0.15)

    ax_bs.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax_bs.xaxis.set_minor_locator(ticker.MultipleLocator(0.01))

    for i, row in transmission_df.iterrows():
        row_id = row['ROW']
        sample_id = row['Sample ID']
        lp = 5 * round(row['Laser power [MW/m2]'] / 5)
        bsc = row['Big spheres']
        ssc = 100. - bsc
        bkt = row['Baking time (min)']
        deposition_rate = row['Deposition rate (nm/s)']
        tracking_at_row_df = tracking_df[tracking_df['ROW'] == row_id]
        tracking_at_row_df.reset_index(inplace=True, drop=True)
        info_txt = f'{lp:.0f} ' + r'MW/m$^{\mathregular{2}}$' + '\n'
        info_txt += r'0.285 $\mathregular{\mu}$m spheres: ' + f'{ssc:.0f} %\n'
        info_txt += r'0.850 $\mathregular{\mu}$m spheres: ' + f'{bsc:.0f} %' + '\n'
        info_txt += f'900 °C, {bkt:.0f} min'
        output_file_tag = f'{lp:.0f}MWpm2_{ssc:03.0f}S_{bsc:03.0f}B_{bkt:03.0f}m'

        if len(tracking_at_row_df) > 0:
            print(f'Working on {sample_id}, ROW {row_id}')
            # print(tracking_at_row_df)
            folder_images = tracking_at_row_df['Test'][0] + '_images'
            path_to_images = os.path.join(path_to_tracking_data, folder_images)
            params = get_experiment_params(path_to_tracking_data, tracking_at_row_df['Test'][0])
            exposure_time = float(params['Camera exposure time']['value'])
            pulse_length = float(params['Emission Time']['value'])
            # print(f'Exposure time: {exposure_time}')
            calibration_csv = f'calibration_20231010_{exposure_time:.0f}_us.csv'
            temperature_calibration = load_calibration(os.path.join(calibration_path, calibration_csv))
            # print(tracking_at_row_df)
            fig, ax = plt.subplots(1, 1, constrained_layout=True)
            fig.set_size_inches(4., 3.)
            ax.set_xlabel('t [s]')
            ax.set_ylabel('T [K]')
            ax.set_ylim(1000, 3500)

            ax.set_title('Pebble temperature')

            max_lines = 8
            colors = plt.cm.cool(np.linspace(0, 1, max_lines))
            current_line = 0
            max_t = -1
            frame_max = -1
            n_points = -1

            max_bs_plots = 4
            cnt_s_plots = 0
            cnt_b_plots = 0

            for j, row_pebble in tracking_at_row_df.iterrows():
                if current_line >= max_lines:
                    break
                pebble_csv = os.path.splitext(row_pebble['Analysis file'])[0] + '_temperature.csv'
                pebble_csv = os.path.join(path_to_images, 'tracking', pebble_csv)
                # print(f'Analysis file: {pebble_csv}')
                pebble_temperature_df = pd.read_csv(pebble_csv, comment='#').apply(pd.to_numeric)
                if pebble_temperature_df['Time (s)'].max() > max_t:
                    max_t = pebble_temperature_df['Time (s)'][j]
                    frame_max = j
                ax.plot(pebble_temperature_df['Time (s)'], pebble_temperature_df['Temperature (K)'],
                        color=colors[current_line])

                if row_id == row_id_s and pebble_temperature_df['Time (s)'].max() < 0.15 and cnt_s_plots < max_bs_plots:
                    ax_bs.plot(pebble_temperature_df['Time (s)'], pebble_temperature_df['Temperature (K)'],
                        color='tab:blue', alpha=0.75)
                    ax_bs.text(
                        0.95, 0.15, r'0.3 mm spheres', va='bottom', ha='right',
                        fontsize=11, color='tab:blue',
                        transform=ax_bs.transAxes,
                    )
                    cnt_s_plots += 1
                if row_id == row_id_b and cnt_b_plots < max_bs_plots: # and pebble_temperature_df['Time (s)'].max() < 0.15:
                    ax_bs.plot(pebble_temperature_df['Time (s)'], pebble_temperature_df['Temperature (K)'],
                        color='tab:purple', alpha=0.75)
                    ax_bs.text(
                        0.95, 0.05, r'0.9 mm spheres', va='bottom', ha='right',
                        fontsize=11, color='tab:purple',
                        transform=ax_bs.transAxes,
                    )
                    cnt_b_plots += 1

                current_line += 1
            ax.set_xlim(0, 0.05 * np.ceil(max_t*1.5 / 0.05))
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.01))
            ax.text(
                0.95, 0.05, info_txt, va='bottom', ha='right', fontsize=11, color='tab:blue',
                transform=ax.transAxes,
            )
            fig.savefig(os.path.join(base_path, output_file_tag + '.png'), dpi=600)
            fig.savefig(os.path.join(base_path, output_file_tag + '.pdf'), dpi=600)
            fig.savefig(os.path.join(base_path, output_file_tag + '.eps'), dpi=600)
            plt.close(fig)

            # Get the image with the highest average gray value
            img_list = [fn for fn in os.listdir(path_to_images) if fn.endswith('.jpg')]
            mean_gray = -1
            image_file = img_list[0]
            for fn in img_list:
                img = cv2.imread(os.path.join(path_to_images, fn), 0)
                mg = img.mean()
                if mg > mean_gray:
                    mean_gray = mg
                    image_file = fn


            # get the temperature over the whole image
            sid = sample_id
            if sample_id == 'R4N127-1':
                sid = 'R4N127'
            img_file_prefix = f'{sid}_ROW{row_id}_IMG-{frame_max}'
            try:
                # image_file = find_file_that_starts_with(
                #     relative_path=path_to_images, starts_with=img_file_prefix
                # )

                img = cv2.imread(os.path.join(path_to_images, image_file), 0)
                if crop_image:
                    img = get_cropped_image(img)
                    # print(f'Cropped image has the following size:')
                    # frameSize = img.shape
                    # print(frameSize)
                temp_im = convert_to_temperature(img, temperature_calibration, max_t, pulse_length, deposition_rate)
                frameSize = img.shape

                fig, ax = plt.subplots(ncols=1, nrows=1, constrained_layout=True)  # , frameon=False)
                # ax.margins(x=0, y=0)
                # plt.autoscale(tight=True)
                scale_factor = 3.5 if crop_image else 1.5
                aspect_ratio = 1.75 if crop_image else 1.5
                w, h = frameSize[1] * scale_factor * aspect_ratio * px2mm / 25.4, frameSize[
                    0] * scale_factor * px2mm / 25.4
                fig.set_size_inches(w, h)

                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="7%", pad=0.025)

                cs = ax.imshow(temp_im, interpolation='none', norm=norm1)  # ,
                # extent=(0, frameSize[1] * px2mm, 0, frameSize[0] * px2mm))
                cbar = fig.colorbar(cs, cax=cax, extend='both')
                cbar.set_label('Temperature (K)\n', size=9, labelpad=9)
                cbar.ax.set_ylim(1500, 3200)
                cbar.ax.ticklabel_format(axis='x', style='sci', useMathText=True)
                cbar.ax.tick_params(labelsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
                cbar.update_ticks()

                gs_rate, area_hot = sublimation_rate_thermal_img(t_img=temp_im, threshold=temp_im.max() * 0.95)
                print(f'Sublimation rate: {gs_rate:.3E} Torr-L/s/m2')
                transmission_df.loc[i, 'Graphite sublimation (Torr-L/s/m^2)'] = gs_rate

                info_img_txt = info_txt + f'\n{gs_rate:.0f} Torr-L/s/m$^{{\mathregular{{2}}}}$'

                time_txt = ax.text(
                    0.95, 0.95, info_img_txt,
                    horizontalalignment='right',
                    verticalalignment='top',
                    color='w',
                    transform=ax.transAxes,
                    fontsize=8
                )

                fig2, ax2 = plt.subplots(ncols=1, nrows=1, constrained_layout=True)  # , frameon=False)
                w, h = frameSize[1] * scale_factor * aspect_ratio * px2mm / 25.4, frameSize[
                    0] * scale_factor * px2mm / 25.4
                fig2.set_size_inches(w, h)
                cs2 = ax2.imshow(threshold_image(temp_im, threshold=temp_im.max() * 0.95), interpolation='none',
                                 norm=norm1)
                divider2 = make_axes_locatable(ax2)
                cax2 = divider2.append_axes("right", size="7%", pad=0.025)

                # extent=(0, frameSize[1] * px2mm, 0, frameSize[0] * px2mm))
                cbar2 = fig2.colorbar(cs2, cax=cax2, extend='both')
                cbar2.set_label('Temperature (K)\n', size=9, labelpad=9)
                cbar2.ax.set_ylim(1500, 3200)
                cbar2.ax.ticklabel_format(axis='x', style='sci', useMathText=True)
                cbar2.ax.tick_params(labelsize=9)
                ax2.set_xticks([])
                ax2.set_yticks([])
                cbar2.update_ticks()

                hot_spot_txt = ax2.text(
                    0.05, 0.05, f'Area hot spots:\n{area_hot:.2f} cm$^{{\mathregular{{2}}}}$',
                    horizontalalignment='left',
                    verticalalignment='bottom',
                    color='w',
                    transform=ax2.transAxes,
                    fontsize=11
                )


                fig.savefig(os.path.join(base_path, os.path.splitext(image_file)[0] + '_temp.png'), dpi=600)
                fig.savefig(os.path.join(base_path, os.path.splitext(image_file)[0] + '_temp.svg'), dpi=600)
                fig.savefig(os.path.join(base_path, os.path.splitext(image_file)[0] + '_temp.pdf'), dpi=600)
                fig.savefig(os.path.join(base_path, os.path.splitext(image_file)[0] + '_temp.eps'), dpi=600)

                fig2.savefig(os.path.join(base_path, os.path.splitext(image_file)[0] + 'hot_spots_.png'), dpi=600)


                plt.close(fig)
            except IndexError as e:
                print(img_file_prefix)
                raise e

            # plt.show()

    fig_bs.savefig(os.path.join(base_path, 'temperature_vs_time_sphere_sizes.png'), dpi=600)
    fig_bs.savefig(os.path.join(base_path, 'temperature_vs_time_sphere_sizes.pdf'), dpi=600)
    fig_bs.savefig(os.path.join(base_path, 'temperature_vs_time_sphere_sizes.eps'), dpi=600)
    plt.close(fig_bs)

    ouput_deposition_rates_csv = os.path.splitext(deposition_rates_csv)[0] + '_graphite_sublimation.csv'
    transmission_df.to_csv(ouput_deposition_rates_csv, index=False)


if __name__ == '__main__':
    main()
