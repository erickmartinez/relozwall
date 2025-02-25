import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import os
import json
from data_processing.utils import get_experiment_params

def map_laser_power_settings(rdir):
    file_list = [fn for fn in os.listdir(rdir) if fn.endswith('.csv')]
    mapping = {}
    for i, f in enumerate(file_list):
        params = get_experiment_params(relative_path=rdir, filename=os.path.splitext(f)[0])
        laser_setpoint = int(params['Laser power setpoint']['value'])
        df = pd.read_csv(os.path.join(rdir, f), comment='#').apply(pd.to_numeric)
        laser_power = df['Laser output peak power (W)'].values
        laser_power = laser_power[laser_power > 0.0]
        mapping[laser_setpoint] = np.round(laser_power.mean(),2)

    keys = list(mapping.keys())
    keys.sort()
    return {i: mapping[i] for i in keys}


def main():
    laser_power_mapping = map_laser_power_settings('data')
    laser_power_mapping_df = pd.DataFrame(data={
        'Laser power setting (%)': np.array([int(k) for k in laser_power_mapping.keys()]),
        'Laser power (W)': np.array([p for p in laser_power_mapping.values()])
    })
    laser_power_mapping_df.set_index(keys=['Laser power setting (%)'], inplace=True)

    print(laser_power_mapping_df)
    laser_power_mapping_df.to_csv(path_or_buf='laser_power_mapping.csv', encoding='utf-8-sig')


if __name__ == '__main__':
    main()
