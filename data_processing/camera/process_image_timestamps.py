"""
This code scans the directory where the images from a 'laser_test_flir.py' experiment are stored and creates
a csv containing the filename, sample_id, row_id, frame_id, timestamp, and time.
"""
import os
import re
import pandas as pd
from tkinter.filedialog import askdirectory
import numpy as np


def main():
    # Get the path where the images are stored
    base_dir = askdirectory(
        title="Select the directory where the images are stored",
        initialdir=r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests'
    )
    # Save results one level up
    save_dir = os.path.dirname(base_dir)
    # Get the list of jpg images in the directory
    file_list = [f for f in os.listdir(base_dir) if f.endswith('.jpg')]
    # The regexp pattern that scans for sample id, row id, frame id and timestamp
    p = re.compile(r'(\w+\d+\d+)\_ROW(\d+)\_IMG\-(\d+)\-(\d+)\.jpg')
    columns = ['sample_id', 'row_id', 'frame_id', 'timestamp', 'time (s)', 'filename']
    df = pd.DataFrame(columns=columns)
    df.set_index(['row_id', 'frame_id'], inplace=True)
    # df['timestamp'] = df['timestamp'].astype('u8')
    t0 = np.inf
    for i, f in enumerate(file_list):
        m = p.match(f)
        sample_id = m.group(1)
        row_id = int(m.group(2))
        frame_id = int(m.group(3))
        timestamp = int(m.group(4))
        # if i == 0:
        t0 = min(t0, timestamp)
        # t = (timestamp - t0) * 1E-9
        print(f"SAMPLE ID: {sample_id:>7}, ROW ID: {row_id:>4d}, FRAME ID: {frame_id:>4d}, t: {t0:>4.3f} (s)")
        row_df = pd.DataFrame(data={
            'sample_id': [sample_id],
            'row_id': [row_id],
            'frame_id': [frame_id],
            'timestamp': [timestamp],
            'time (s)': [timestamp],
            'filename': [f]
        })
        row_df.set_index(['row_id', 'frame_id'], inplace=True)
        # row_df['timestamp'] = row_df['timestamp'].astype('str')
        # df = df.append(row_df)
        df = pd.concat([df, row_df])
    # df[['row_id', 'frame_id', 'time (s)']] = df[['row_id', 'frame_id', 'time (s)']].apply(pd.to_numeric)
    df['time (s)'] = (df['timestamp'] - t0) * 1E-9
    df.sort_values(by=['row_id', 'frame_id'], inplace=True)
    print(df[df.columns[0:-1]])
    # df['timestamp'] = df['timestamp'].astype('str')
    csv_file = os.path.join(save_dir, base_dir + '_time.csv')
    df.to_csv(path_or_buf=csv_file)


if __name__ == '__main__':
    main()
