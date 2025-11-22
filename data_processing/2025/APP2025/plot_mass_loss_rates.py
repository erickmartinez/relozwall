import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from data_processing.misc_utils.plot_style import load_plot_style
import os
import re
from scipy.interpolate import interp1d
from typing import Tuple, Dict, Union

SHOTS = [203782, 203783, 203784]

PATH_TO_EVAPORATION_RATES = r'./data/d3d_evaporation_rates'
PATH_TO_MASS_LOSS_RATES = r'./data/mass_loss_rate_model/model_results'
PATH_TO_MDS_EMISSION_RATS = r'./data/mds_spectra/emission_flux'

def load_evaporation_rates(shot, path_to_data=PATH_TO_EVAPORATION_RATES):
    path_to_data = Path(path_to_data)
    path_to_csv = path_to_data / f'{shot}_evaporation_rate.csv'
    df = pd.read_csv(path_to_csv, comment='#').apply(pd.to_numeric, errors='coerce')
    time = df['Time (s)'].values
    evaporation_rate = df['Evaporation rate (atoms/s)'].values #
    evaporation_rate_lb = df['Evaporation rate lb (atoms/s)'].values
    evaporation_rate_ub = df['Evaporation rate ub (atoms/s)'].values

    data = {
        'time (s)': time,
        'evaporation_rate (atoms/s)': evaporation_rate,
        'evaporation_rate_lb (atoms/s)': evaporation_rate_lb,
        'evaporation_rate_ub (atoms/s)': evaporation_rate_ub,
    }

    return data

def load_mass_loss_rates(shot, path_to_data=PATH_TO_MASS_LOSS_RATES) -> Dict[str, np.ndarray]:
    path_to_data = Path(path_to_data)
    if shot in [203780, 203781]:
        path_to_h5 = path_to_data / f'203780-203781_mass_loss_model.h5'
    elif shot in [203782, 203783, 203784]:
        path_to_h5 = path_to_data / f'203782-203784_mass_loss_model.h5'
    else:
        raise ValueError(f'Shot {shot} not supported.')

    with h5py.File(path_to_h5, 'r') as h5:
        shot_gp = h5[f'/{shot}']
        time_model = np.array(shot_gp.get('time'))
        mass_loss_rate = np.array(shot_gp.get('mass_loss_rate'))
        mass_loss_rate_error = np.array(shot_gp.get('mass_loss_error'))
        qpara = np.array(shot_gp.get('qpara'))

    data = {
        'time (s)': time_model,
        'mass loss_rate (atoms/s)': mass_loss_rate,
        'mass loss rate error (atoms/s)': mass_loss_rate_error,
        'qpara (MW/m2)': qpara,
    }

    return data

def load_mds_emission(shot, path_to_data=PATH_TO_MDS_EMISSION_RATS) -> Dict[str, np.ndarray]:
    path_to_data = Path(path_to_data)
    files_in_dir = [fn for fn in os.listdir(str(path_to_data)) if fn.startswith(f'{shot}_') and fn.endswith('.csv')]
    path_to_file = path_to_data / files_in_dir[0]
    pattern = re.compile(fr'{shot}_emission_flux_(.*?).csv')
    match = re.search(pattern, files_in_dir[0])
    if not(match):
        raise ValueError(f'Shot {shot} not found.')

    emission_line = match.group(1).replace('-', '')
    df = pd.read_csv(path_to_file, comment='#').apply(pd.to_numeric, errors='coerce')
    time = df['time (s)'].values
    flux = df[f'Flux {emission_line} (molecules/s)'].values

    data = {
        'time (s)': time,
        'flux (molecules/s)': flux,
        'emission line': emission_line,
    }

    return data


def main(shots):
    n_shots = len(shots)
    load_plot_style()
    fig, axes = plt.subplots(nrows=n_shots, sharex=True, sharey=True, constrained_layout=True)
    fig.set_size_inches(4.5, 8)

    for i, shot in enumerate(shots):
        mass_loss_data = None
        mds_data = None
        evaporation_data = None

        try:
            mass_loss_data = load_mass_loss_rates(shot)
        except Exception as e:
            mass_loss_data = None
            print(f'Error loading mass loss data from {shot}: {e}')

        try:
            mds_data = load_mds_emission(shot)
        except Exception as e:
            print(f'Error loading mds emission data from {shot}: {e}')
            mds_data = None

        try:
            evaporation_data = load_evaporation_rates(shot)
        except Exception as e:
            print(f'Error loading evaporation data from {shot}: {e}')

        if mass_loss_data is not None:
            line_ml,  = axes[i].plot(
                mass_loss_data['time (s)'], mass_loss_data['mass loss_rate (atoms/s)'],
                color='C0', ls='-', label='Total mass loss'
            )

            y1 = mass_loss_data['mass loss_rate (atoms/s)'] - mass_loss_data['mass loss rate error (atoms/s)']
            y2 = mass_loss_data['mass loss_rate (atoms/s)'] + mass_loss_data['mass loss rate error (atoms/s)']
            axes[i].fill_between(
                mass_loss_data['time (s)'],
                y1,
                y2,
                ls='-', color='C0', alpha=0.3
            )


        if mds_data is not None:
            line_mds, = axes[i].plot(
                mds_data['time (s)'], mds_data['flux (molecules/s)'], color='C1', ls='-',
                label=f'{mds_data["emission line"]}'
            )

        if evaporation_data is not None:
            line_e, = axes[i].plot(
                evaporation_data['time (s)'], evaporation_data['evaporation_rate (atoms/s)'],
                color='C2', ls='-', label='Evaporation'
            )

            axes[i].fill_between(
                evaporation_data['time (s)'],
                evaporation_data['evaporation_rate_lb (atoms/s)'],
                evaporation_data['evaporation_rate_ub (atoms/s)'],
                color='C2', alpha=0.3
            )

        t_min, t_max = 1.5, 3.
        time_mass_loss = mass_loss_data['time (s)']
        msk_mass_loss = (t_min <= time_mass_loss) & (time_mass_loss <= t_max)

        path_to_compiled_data = Path(r'./data/mass_loss_rates_compiled')
        path_to_compiled_data.mkdir(parents=True, exist_ok=True)
        path_to_h5 = path_to_compiled_data / f'{shot}_mass_loss_rates.h5'

        with h5py.File(str(path_to_h5), 'w') as hf:
            shot_group = hf.require_group(f'{shot}')
            time_ds = shot_group.create_dataset('time', data=mass_loss_data['time (s)'][msk_mass_loss], compression='gzip')
            time_ds.attrs['units'] = 's'
            qpara_ds = shot_group.create_dataset('qpara', data=mass_loss_data['qpara (MW/m2)'][msk_mass_loss], compression='gzip')
            qpara_ds.attrs['units'] = 'MW/m2'
            mass_loss_ds = shot_group.create_dataset('mass_loss_rate', data=mass_loss_data['mass loss_rate (atoms/s)'][msk_mass_loss], compression='gzip')
            mass_loss_ds.attrs['units'] = 'atoms/s'
            mass_loss_delta_ds = shot_group.create_dataset('mass_loss_rate_error', data=mass_loss_data['mass loss rate error (atoms/s)'][msk_mass_loss], compression='gzip')
            mass_loss_delta_ds.attrs['units'] = 'atoms/s'

        # Try interpolating mds data
        try:
            msk_mds_data = (t_min <= mds_data['time (s)']) & (mds_data['time (s)'] <= t_max)
            time_mds = mds_data['time (s)'][msk_mds_data]
            spectral_line_emission = mds_data['flux (molecules/s)'][msk_mds_data]
            interp_line_emission =interp1d(time_mds, spectral_line_emission, kind='cubic', fill_value='extrapolate')
            with h5py.File(str(path_to_h5), 'r+') as hf:
                shot_group = hf.require_group(f'{shot}')
                mds_line_emission_ds = shot_group.create_dataset(f'{mds_data["emission_line"]}_flux', data=interp_line_emission(time_mass_loss), compression='gzip')
                mds_line_emission_ds.attrs['units'] = 'atoms/s'



        except Exception as e:
            print(f'Error loading mds data from {shot}: {e}')

        # Try interpolating evaporation data
        try:
            time_evaporation = evaporation_data['time (s)']
            msk_evaporation = (t_min <= time_evaporation) & (time_evaporation <= t_max)
            time_evaporation = time_evaporation[msk_evaporation]
            evaporation_rate = evaporation_data['evaporation_rate (atoms/s)'][msk_evaporation]
            evaporation_rate_lb = evaporation_data['evaporation_rate_lb (atoms/s)'][msk_evaporation]
            evaporation_rate_ub = evaporation_data['evaporation_rate_ub (atoms/s)'][msk_evaporation]

            interp_evaporation = interp1d(x=time_evaporation, y=evaporation_rate, fill_value='extrapolate')
            interp_evaporation_lb = interp1d(x=time_evaporation, y=evaporation_rate_lb, fill_value='extrapolate')
            interp_evaporation_ub = interp1d(x=time_evaporation, y=evaporation_rate_ub, fill_value='extrapolate')



            with h5py.File(str(path_to_h5), 'r+') as hf:
                shot_group = hf.require_group(f'{shot}')
                evaporation_rate_ds = shot_group.create_dataset('evaporation_rate', data=interp_evaporation(time_evaporation), compression='gzip')
                evaporation_rate_ds.attrs['units'] = 'atoms/s'
                evaporation_rate_lb_ds = shot_group.create_dataset('evaporation_rate_lb', data=interp_evaporation_lb(time_evaporation), compression='gzip')
                evaporation_rate_ub_ds = shot_group.create_dataset('evaporation_rate_ub', data=interp_evaporation_ub(time_evaporation), compression='gzip')
                evaporation_rate_ub_ds.attrs['units'] = 'atoms/s'
                evaporation_rate_ub_ds.attrs['units'] = 'atoms/s'
        except Exception as e:
            print(f'Error loading evaporation data from {shot}: {e}')





        axes[i].set_title(f'Shot #{shot}')
        axes[i].set_yscale('log')
        axes[i].set_ylim(1E14, 1E24)
        axes[i].legend(ncol=3, loc='upper left', fontsize='9')


    fig.supylabel(f'Mass loss rate (atoms/s)')
    axes[-1].set_xlabel('Time (s)')
    axes[-1].set_xlim(1.5, 3.5)

    path_to_figures = Path(r'./figures')
    path_to_figures.mkdir(parents=True, exist_ok=True)
    for extension in ['png', 'pdf', 'svg']:
        path_to_figure = path_to_figures / f'{shot}.{extension}'
        fig.savefig(path_to_figure, dpi=600, bbox_inches='tight')


    plt.show()




if __name__ == '__main__':
    main(shots=SHOTS)



