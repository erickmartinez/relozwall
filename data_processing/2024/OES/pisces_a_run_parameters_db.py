import os

import pandas as pd
import numpy as np
import re
from datetime import datetime

f_pattern = re.compile(r'(.*)?\:\s+(.*)')
d_pattern = '%a %b %d %H:%M:%S %Y'

# Function to read
# last N lines of the file
def LastNlines(fname, N):
    # assert statement check
    # a condition
    assert N >= 0

    # declaring variable
    # to implement
    # exponential search
    pos = N + 1

    # list to store
    # last N lines
    lines = []

    # opening file using with() method
    # so that file get closed
    # after completing work
    with open(fname) as f:

        # loop which runs
        # until size of list
        # becomes equal to N
        while len(lines) <= N:

            # try block
            try:
                # moving cursor from
                # left side to
                # pos line from end
                f.seek(-pos, 2)

            # exception block
            # to handle any run
            # time error
            except IOError:
                f.seek(0)
                break

            # finally block
            # to add lines
            # to list after
            # each iteration
            finally:
                lines = list(f)

            # increasing value
            # of variable
            # exponentially
            pos *= 2

    # returning the
    # whole list
    # which stores last
    # N lines
    return lines[-N:]

def get_echelle_params(path_to_file):
    # with open(path_to_file, 'r') as fp:
    #     for count, line in enumerate(fp):
    #         pass

    footer = LastNlines(path_to_file, 40)
    params = {}
    for line in footer:
        matches = f_pattern.match(line)
        if not matches is None:
            params[matches.group(1)] = matches.group(2)
    return params

def main():
    path_to_runs = './data/Echelle_data'
    paths = os.listdir(path_to_runs)

    map_cols = {
        'Timestamp': 'Date and Time', 'Data Type': 'Data Type', 'Acquisition Mode': 'Acquisition Mode',
        'Exposure Time (secs)': 'Exposure Time (secs)', 'Number of Accumulations': 'Number of Accumulations',
        'Horizontal binning': 'Horizontal binning',
        'Vertical Shift Speed (usecs)': 'Vertical Shift Speed (usecs)',
        'Pixel Readout Rate (MHz)': 'Pixel Readout Rate (MHz)', 'Pre-Amplifier Gain': 'Pre-Amplifier Gain',
        'Gain level': 'Gain level', 'Gate Width (nsecs)': 'Gate Width (nsecs)',
        'Gate Delay (nsecs)': 'Gate Delay (nsecs)', 'Current Temperature (C)': 'Current Temperature'
    }

    db_df = pd.DataFrame(data={
        'Folder': [], 'File': [], 'Timestamp': [], 'Data Type': [], 'Acquisition Mode': [],
        'Exposure Time (secs)': [], 'Number of Accumulations': [],
        'Horizontal binning': [], 'Vertical Shift Speed (usecs)': [],
        'Pixel Readout Rate (MHz)': [], 'Pre-Amplifier Gain': [], 'Gain level': [],
        'Gate Width (nsecs)': [], 'Gate Delay (nsecs)': [], 'Current Temperature (C)': []
    })

    numeric_cols = [
        'Exposure Time (secs)', 'Number of Accumulations', 'Horizontal binning',
        'Vertical Shift Speed (usecs)',
        'Pixel Readout Rate (MHz)', 'Pre-Amplifier Gain', 'Gain level', 'Gate Width (nsecs)',
        'Gate Delay (nsecs)', 'Current Temperature (C)'
    ]


    for rpath in paths:
        file_list = [f for f in os.listdir(os.path.join(path_to_runs, rpath)) if f.endswith('.asc')]
        for file in file_list:
            path_to_file = os.path.join(path_to_runs, rpath, file)
            params = get_echelle_params(path_to_file=path_to_file)
            param_data = {}
            for col in map_cols:
                try:
                    key = map_cols[col]
                    param_data[col] = [params[key]]
                    if col == 'Timestamp':
                        ts = params[key]
                        param_data[col] = [datetime.strptime(params[key], '%a %b %d %H:%M:%S %Y')]
                except KeyError as ke:
                    print(f'Error reading: {path_to_file}')
                    param_data[col] = 0
                    if col == 'Number of Accumulations':
                        param_data[col] = 1

            param_data['Folder'] = rpath
            param_data['File'] = file
            row = pd.DataFrame(data=param_data)
            db_df = pd.concat([db_df, row], axis=0, ignore_index=True)


    db_df[numeric_cols] = db_df[numeric_cols].apply(pd.to_numeric)
    db_df.sort_values(by=['Folder', 'File'], ascending=[True,True], inplace=True)
    print(db_df)
    db_df.to_excel('./data/echelle_db_(programmatic).xlsx', sheet_name='Spectrometer parameters', index=False)





if __name__ == '__main__':
    main()