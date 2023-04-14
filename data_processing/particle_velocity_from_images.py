import json
import os
import re
from scipy import stats as st
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

base_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\SS_TUBE\GC'
info_csv = r'LCT_R4N55_100PCT_2023-03-16_1.csv'
tracking_points_csv = r'LCT_R4N55_100PCT_2023-03-16_1_trackpoints.csv'
frame_rate = 199.
pixel_size = 20.1  # pixels/mm
p = re.compile(r'.*?-(\d+)\.jpg')
nmax = 200
calibration_csv = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\thermal camera\calibration\adc_calibration_curve.csv'
px2mm = 1. / pixel_size
parallax_span = 30. # deg
camera_angle = 18.  # deg

def load_tracking_data():
    df = pd.read_csv(
        os.path.join(base_path, tracking_points_csv), usecols=['TID', 'PID', 'x [pixel]', 'y [pixel]', 't [sec]']
    )
    return df.apply(pd.to_numeric)


def main():
    trajectories_df = load_tracking_data()
    trajectory_ids = trajectories_df['TID'].unique()
    main_axis_projector_factor = np.sin(camera_angle*np.pi/180.)
    velocities = []
    for tid in trajectory_ids:
        points_df = trajectories_df[trajectories_df['TID'] == tid]
        dx = points_df['x [pixel]'].diff().dropna().values
        # dy = px2mm * points_df['y [pixel]'].diff().dropna().values
        vx = 0.1 * px2mm * np.abs(dx) * frame_rate * main_axis_projector_factor
        print(f'TID: {tid}, points: {len(points_df)}, 1 px = {px2mm:.3} mm, projecting factor: {main_axis_projector_factor:.3f}')
        print(f'dx (pixels)', dx)
        print(f'dx (mm)', dx*px2mm)
        print(f'dx (cm)', dx * px2mm * 0.1)
        print(f'vx (cm/s)', vx)
        if len(vx) > 0:
            velocities.append(vx.mean())
        # for v in vx:
        #     velocities.append(v)
    velocities = np.array(velocities)
    mean_velocity = velocities.mean()
    print(velocities)
    print(f'Mean velocity: {mean_velocity:.1f} cm/s')
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    fig, ax = plt.subplots(ncols=1, nrows=1, constrained_layout=True, frameon=False)
    fig.set_size_inches(4., 3.)

    ax.hist(velocities, bins=20)

    ax.set_xlabel('$v_z$ (cm/s)')
    ax.set_ylabel('Counts')
    ax.set_title('Pebble ejection velocity')
    ax.set_xlim(0, 200)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    results_txt = f'Average: {mean_velocity:.1f} cm/s\nMode: {st.mode(velocities).mode[0]:.1f} cm/s'
    ax.text(
        0.95,
        0.95,
        results_txt,
        fontsize=9,
        # color='tab:green',
        transform=ax.transAxes,
        va='top', ha='right',
        bbox=props
    )

    fig.savefig(os.path.join(base_path, os.path.splitext(info_csv)[0] + '_velocity_histogram.png'), dpi=600)
    plt.show()




if __name__ == '__main__':
    main()
