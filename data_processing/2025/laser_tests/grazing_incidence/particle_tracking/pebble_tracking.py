import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pims
import trackpy as tp
from data_processing.misc_utils.plot_style import load_plot_style
from pathlib import Path
from tkinter import filedialog
import os
import re

PATH_TO_IMAGES = r'/Users/erickmartinez/Library/CloudStorage/OneDrive-Personal/Documents/ucsd/Research/Data/2025/laser_tests/GRAZING_INCIDENCE/LCT_R5N16-0912_100PCT_2025-09-15_1_images'

def get_path_to_images():
    folder_path = filedialog.askdirectory()
    return Path(folder_path)

def get_image_paths_and_times(path_to_images):
    files = [fn for fn in os.listdir(path_to_images) if fn.endswith('.tiff')]
    results = np.empty(len(files), dtype=np.dtype([('frame', 'i'), ('time (s)', 'd'), ('filename', '<U100')]))
    p = re.compile(r'.*\_IMG\-(\d+)\-(\d+).tiff')

    for i, f in enumerate(files):
        results[i] = os.path.getmtime(os.path.join(path_to_images, f))
        m = p.match(f)
        if m:
            results['frame'][i] = int(m.group(1))
            results['time (s)'][i] =  float(m.group(2)) * 1E-9
            results['filename'][i] = f
            # print(m.group(1), m.group(2))

    results.sort()
    results['time (s)'] = results['time (s)'] - results['time (s)'][0]
    return results



def main(folder_path=None):
    if folder_path is None:
        folder_path = get_path_to_images()

    # frames = pims.open(f'{folder_path}/*.tiff')
    time_stamps = get_image_paths_and_times(folder_path)
    paths_to_images = [f'{folder_path}/{f["filename"].strip()}' for f in time_stamps]
    # for path in paths_to_images:
    #     print(path)

    frames = [pims.Frame(pims.open(path)) for path in paths_to_images]
    # frames = pims.ReaderSequence(paths_to_images)

    # print(len(frames))




    load_plot_style()
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)

    # ax.imshow(frames[81])
    # f = tp.locate(frames[81], 13, invert=False, minmass=200, maxsize=50)
    # tp.annotate(f, frames[81])
    f = tp.batch(frames[1:150], 13, minmass=200, invert=False, maxsize=50)
    t = tp.link(f, 20, memory=1)

    t1 = tp.filter_stubs(t, 10)
    # Compare the number of particles in the unfiltered and filtered data.
    print('Before:', t['particle'].nunique())
    print('After:', t1['particle'].nunique())

    # tp.mass_size(t1.groupby('particle').mean());  # convenience function -- just plots size vs. mass

    t2 = t1[((t1['mass'] > 50) & (t1['size'] < 15))]
    # tp.annotate(t2[t2['frame'] == 0], frames[0])
    tp.plot_traj(t1)

    plt.show()


if __name__ == '__main__':
    main(folder_path=PATH_TO_IMAGES)
