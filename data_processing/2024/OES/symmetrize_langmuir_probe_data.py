"""
This code symmetrizes the n_e(x), T_e(x) with respect to the peak in n_e(x)
"""
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import json
from scipy.optimize import least_squares, OptimizeResult
from matplotlib.lines import Line2D

lang_probe_results_file = r'./data/PA_probe/20241031/langprobe_results/lang_results_gamma_ivdata0011.csv'


def load_plot_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)
    mpl.rcParams['text.latex.preamble'] = (r'\usepackage{mathptmx}'
                                           r'\usepackage{xcolor}'
                                           r'\usepackage{helvet}'
                                           r'\usepackage{siunitx}'
                                           r'\usepackage{amsmath, array, makecell}')


def on_mouse_move(event, horizontal_lines, vertical_lines, fig):
    # Check if the mouse is inside the plot axes
    if event.inaxes is not None:
        # Make lines visible and update their positions
        for horizontal_line, vertical_line in zip(horizontal_lines, vertical_lines):
            horizontal_line.set_visible(True)
            vertical_line.set_visible(True)
            horizontal_line.set_ydata([event.ydata, event.ydata])  # Changed to list format
            vertical_line.set_xdata([event.xdata, event.xdata])    # Changed to list format
        fig.canvas.draw()  # Use draw() instead of draw_idle()
    else:
        # Hide lines when mouse leaves plot area
        for horizontal_line, vertical_line in zip(horizontal_lines, vertical_lines):
            horizontal_line.set_visible(False)
            vertical_line.set_visible(False)
        fig.canvas.draw()

picked_x, picked_y = 0, 0
def on_click(event, ax):
    global picked_x, picked_y
    # if isinstance(event.artist, Line2D) and event.artist in ax.lines:
    #     thisline = event.artist
    #     xdata = thisline.get_xdata()
    #     ydata = thisline.get_ydata()
    #     ind = event.ind
    #     x_mean = np.mean(xdata[ind])
    #     y_mean = np.mean(ydata[ind])
    #     picked_x, picked_y = x_mean, y_mean
    picked_x = event.xdata
    picked_y = event.ydata


def symmetrize_dataset(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert an asymmetric dataset of (x,y) points around x=0 into a symmetric dataset
    by mirroring all points from both sides to their opposite sides, without duplicating
    points at the same x-coordinates.

    Parameters:
    -----------
    x : np.ndarray
        Array of x coordinates
    y : np.ndarray
        Array of y coordinates

    Returns:
    --------
    tuple[np.ndarray, np.ndarray]
        Tuple containing the symmetric x and y arrays
    """
    # Create a DataFrame for easier manipulation
    df = pd.DataFrame({'x': x, 'y': y}).drop_duplicates('x')

    # Separate points
    zero_points = df[df['x'] == 0].copy()
    left_points = df[df['x'] < 0].copy()
    right_points = df[df['x'] > 0].copy()

    # For each point on the left, ensure there's a corresponding point on the right
    missing_right = pd.DataFrame({
        'x': -left_points['x'].values,
        'y': left_points['y'].values
    })

    # For each point on the right, ensure there's a corresponding point on the left
    missing_left = pd.DataFrame({
        'x': -right_points['x'].values,
        'y': right_points['y'].values
    })

    # Combine all points while removing duplicates
    result_df = pd.concat([
        left_points,
        right_points,
        missing_left[~missing_left['x'].isin(left_points['x'])],
        missing_right[~missing_right['x'].isin(right_points['x'])],
        zero_points
    ])

    # Sort by x coordinate
    result_df = result_df.sort_values('x').drop_duplicates('x')

    return result_df['x'].values, result_df['y'].values

def symmetrize_data(x_values, y_values):
    x_left = x_values.min()
    msk_positive = x_values > 0.
    msk_negative = x_values < 0.
    idx_0 = np.argmin(np.abs(x_values))
    x0 = x_values[idx_0]
    y_0 = y_values[idx_0]
    x_pos, y_pos = x_values[msk_positive], y_values[msk_positive]
    n_pos = len(x_pos)
    x_neg, y_neg = x_values[msk_negative], y_values[msk_negative]
    x_sym_neg = -x_pos[::-1]
    y_sym_neg = np.empty(n_pos, dtype=np.float64)
    idx_right = len(y_neg)
    if idx_right < 1:
        return np.hstack((x_sym_neg, x0, x_pos)), np.hstack((y_pos[::-1], y_0, y_pos))

    x_sym_neg[-idx_right::] = x_neg
    y_sym_neg[-idx_right::] = y_neg
    y_sym_pos = y_pos[idx_right-1:-1]
    y_left = y_sym_pos[::-1]
    y_sym_neg[0:-idx_right] = y_left
    return np.hstack((x_sym_neg, x0, x_pos)), np.hstack((y_sym_neg, y_0, y_pos))


def main():
    global lang_probe_results_file
    global picked_x, picked_y
    rel_path = os.path.dirname(lang_probe_results_file)
    file_tag = os.path.splitext(os.path.basename(lang_probe_results_file))[0]
    folder_symmetrized = os.path.join(rel_path, 'symmetrized')
    if not os.path.exists(folder_symmetrized):
        os.makedirs(folder_symmetrized)
    data_df = pd.read_csv(filepath_or_buffer=lang_probe_results_file).apply(pd.to_numeric)
    data_df.sort_values(by=['x (cm)'], inplace=True)


    x = data_df['x (cm)'].values
    n_e = data_df['n_e (cm^{-3})'].values
    T_e = data_df['T_e (eV)'].values
    V_s = data_df['V_sc (eV)'].values
    J_sat = data_df['J_sat (A/cm^2)'].values
    J_sat_robust = data_df['J_sat_robust (A/cm^2)'].values

    load_plot_style()
    # mpl.use('macosx')
    # ['gtk3agg', 'gtk3cairo', 'gtk4agg', 'gtk4cairo', 'macosx', 'nbagg', 'notebook', 'qtagg', 'qtcairo', 'qt5agg', 'qt5cairo', 'tkagg', 'tkcairo', 'webagg', 'wx', 'wxagg', 'wxcairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template', 'widget', 'ipympl', 'inline']
    # mpl.use('agg')
    # plt.ion()
    fig_pp, axes_pp = plt.subplots(nrows=2, ncols=2, constrained_layout=True, sharex=True)
    fig_pp.set_size_inches(8.0, 6.0)
    line, = axes_pp[0, 0].plot(x, n_e / 1e11, marker='o', color='C0', ls='none', mfc='none', mew=1.25, picker=True, pickradius=5)  # numerical fit
    axes_pp[0, 0].set_xlabel('$x$ {\sffamily (cm)}', usetex=True)
    axes_pp[0, 0].set_ylabel(r'$n_e$ {\sffamily (x 10\textsuperscript{-11} cm\textsuperscript{-3})}', usetex=True)
    axes_pp[0, 0].set_title('Density')

    axes_pp[0, 1].plot(x, T_e, marker='o', color='C0', ls='none', mfc='none', mew=1.25)  # numerical fit
    axes_pp[0, 1].set_xlabel('Position (cm)')
    axes_pp[0, 1].set_ylabel(r'$T_e$ {\sffamily (eV)}', usetex=True)
    axes_pp[0, 1].set_title('Temperature')

    axes_pp[1, 0].plot(x, V_s, marker='o', color='C0', ls='none', mfc='none', mew=1.25)  # numerical fit
    axes_pp[1, 0].set_xlabel('$x$ {\sffamily (cm)}', usetex=True)
    axes_pp[1, 0].set_ylabel(r'$V_{\mathrm{s}}$ {\sffamily (eV)}', usetex=True)
    axes_pp[1, 0].set_title('Space-charge potential')

    axes_pp[1, 1].plot(x, J_sat, marker='o', color='C0', ls='none', mfc='none', mew=1.25)  # numerical fit to Jsat
    axes_pp[1, 1].plot(x, J_sat_robust, marker='o', color='tab:green', ls='none', mfc='none', mew=1.25)  # robust Jsat fit
    axes_pp[1, 1].set_xlabel('$x$ {\sffamily (cm)}', usetex=True)
    axes_pp[1, 1].set_ylabel(r'$J_{\mathrm{sat}}$ {\sffamily (A/cm\textsuperscript{2})}', usetex=True)
    axes_pp[1, 1].set_title('Ion saturation current density')

    fig_pp.canvas.mpl_connect(
        'button_press_event',
        lambda event: on_click(event, axes_pp[0,0])
    )

    # Initialize crosshairs
    horizontal_lines, vertical_lines = [], []
    for ax in axes_pp.flatten():
        horizontal_line = ax.axhline(y=0, color='k', ls='--', alpha=0.5, visible=False)
        vertical_line = ax.axvline(x=0, color='k', ls='--', alpha=0.5, visible=False)
        horizontal_lines.append(horizontal_line)
        vertical_lines.append(vertical_line)
    fig_pp.canvas.mpl_connect('motion_notify_event', lambda event: on_mouse_move(event, horizontal_lines, vertical_lines, fig_pp))

    plt.show(block=False)

    ne_max = data_df['n_e (cm^{-3})'].max()
    # Find the x for the peak n_e
    idx_peak = data_df['n_e (cm^{-3})'] == ne_max
    x_peak = data_df.loc[idx_peak, 'x (cm)'].values[-1]
    n_e_peak = data_df.loc[idx_peak, 'n_e (cm^{-3})'].values[-1]

    # Ask user input to recommend a few indices to the left of the peak to symmetryze the data
    ne_peak_ok = False
    input('Select n_e peak on the plot.')
    while not ne_peak_ok:
        response = input(f"Current value: ({picked_x:.3f}, {picked_y:.3E}). Is it good (y/n)?")
        if response == 'y':
            x_peak = picked_x
            ne_peak_ok = True

    # Ask user input to recommend a few indices to the left of the peak to symmetryze the data
    print(f"Recommend an index below or at the peak to set the lower bound for the dataset to be symmetrized.")
    left_bound_ok = False

    x_left = 0.
    while not left_bound_ok:
        input('Select a point on the n_e plot for the lower bond of x')
        response = input(f"Current value: ({picked_x:.3f}, {picked_y:.3E}). Is it good (y/n)?")
        if response == 'y':
            x_left = picked_x
            left_bound_ok = True

    right_bound_ok = False
    x_right = 0.
    while not right_bound_ok:
        input('Select a point on the n_e plot for the upper bond of x')
        response = input(f"Current value: ({picked_x:.3f}, {picked_y:.3E}). Is it good (y/n)?")
        if response == 'y':
            x_right = picked_x
            right_bound_ok = True

    constrained_df = data_df[data_df['x (cm)'].between(x_left, x_right)].reset_index(drop=True)

    idx_peak = np.argmin(np.abs(x - x_peak))
    x_peak = x[idx_peak]
    n_e_peak = n_e[idx_peak]

    print(f'The peak is located at x={x_peak:.2f} cm, n_e(x_peak) = {n_e_peak:.2E} 1/cm^3')

    constrained_df['x_centered (cm)'] = constrained_df['x (cm)'] - x_peak
    x = constrained_df['x_centered (cm)'].values

    n_e = constrained_df['n_e (cm^{-3})'].values
    T_e = constrained_df['T_e (eV)'].values
    V_s = constrained_df['V_sc (eV)'].values
    J_sat = constrained_df['J_sat (A/cm^2)'].values

    x_err = constrained_df['x error (cm)'].values
    n_e_err = constrained_df['n_e error (cm^{-3})'].values
    T_e_err = constrained_df['T_e error (eV)'].values
    V_s_err = constrained_df['V_sc error (eV)'].values
    J_sat_err = constrained_df['J_sat error (A/cm^2)'].values

    x_sym, n_e_sym = symmetrize_dataset(x, n_e)
    synthetic_msk = x_sym < x.min()

    _, T_e_sym = symmetrize_dataset(x, T_e)
    _, J_sat_sym = symmetrize_dataset(x, J_sat)
    _, V_s_sym = symmetrize_dataset(x, V_s)

    # Symmetryze the uncertainties
    _, x_e_err_sym = symmetrize_dataset(x, x_err)
    _, n_e_err_sym = symmetrize_dataset(x, n_e_err)
    _, T_e_err_sym = symmetrize_dataset(x, T_e_err)
    _, J_sat_err_sym = symmetrize_dataset(x, J_sat_err)
    _, V_s_err_sym = symmetrize_dataset(x, V_s_err)



    fig, axes = plt.subplots(2, 2, layout='constrained', sharex=True)
    fig.set_size_inches(7.0, 6.5)

    axes[0,0].plot(x, n_e/1E11, marker='o', ms=6, mfc='none', mew=1.25, ls='none', color='C0')
    axes[0,0].plot(x_sym[synthetic_msk], n_e_sym[synthetic_msk]/1E11, marker='x', ms=6, mfc='none', mew=1.25, ls='none', color='red')
    axes[0,0].set_xlabel('$x$ {\sffamily (cm)}', usetex=True)
    axes[0,0].set_ylabel(r'$n_e$ {\sffamily (x 10\textsuperscript{-11} cm\textsuperscript{-3})}', usetex=True)
    axes[0,0].set_title('Density')

    axes[0, 1].plot(x, T_e, marker='s', ms=6, mfc='none', mew=1.25, ls='none', color='C1')
    axes[0, 1].plot(x_sym[synthetic_msk], T_e_sym[synthetic_msk], marker='x', ms=6, mfc='none', mew=1.25,
                    ls='none', color='red')
    axes[0, 1].set_xlabel('$x$ {\sffamily (cm)}', usetex=True)
    axes[0, 1].set_ylabel(r'$T_e$ {\sffamily (eV)}', usetex=True)
    axes[0, 1].set_title('Temperature')

    axes[1, 0].plot(x, V_s, marker='^', ms=6, mfc='none', mew=1.25, ls='none', color='C1')
    axes[1, 0].plot(x_sym[synthetic_msk], V_s_sym[synthetic_msk], marker='x', ms=6, mfc='none', mew=1.25,
                    ls='none', color='red')
    axes[1, 0].set_xlabel('$x$ {\sffamily (cm)}', usetex=True)
    axes[1, 0].set_ylabel(r'$V_{\mathrm{s}}$ {\sffamily (eV)}', usetex=True)
    axes[1, 0].set_title('Space-charge potential')

    axes[1, 1].plot(x, J_sat, marker='D', ms=6, mfc='none', mew=1.25, ls='none', color='C1')
    axes[1, 1].plot(x_sym[synthetic_msk], J_sat_sym[synthetic_msk], marker='x', ms=6, mfc='none', mew=1.25,
                    ls='none', color='red')
    axes[1, 1].set_xlabel('$x$ {\sffamily (cm)}', usetex=True)
    axes[1, 1].set_ylabel(r'$J_{\mathrm{sat}}$ {\sffamily (A/cm\textsuperscript{2})}', usetex=True)
    axes[1, 1].set_title('Ion saturation current density')

    plt.show()


    symmetrized_df = pd.DataFrame(data={
        'x (cm)': x_sym,
        'x error (cm)': x_e_err_sym,
        'n_e (cm^{-3})': n_e_sym,
        'n_e error (cm^{-3})': n_e_err_sym,
        'T_e (eV)': T_e_sym,
        'T_e error (eV)': T_e_err_sym,
        'J_sat (A/cm^2)': J_sat_sym,
        'J_sat error (A/cm^2)': J_sat_err_sym,
        'V_sc (eV)': V_s_sym,
        'V_sc error (eV)': V_s_err_sym,
        'synthetic': synthetic_msk
    })


    symmetrized_df.to_csv(os.path.join(folder_symmetrized, file_tag + '_symmetrized.csv'), index=False)




if __name__ == '__main__':
    main()