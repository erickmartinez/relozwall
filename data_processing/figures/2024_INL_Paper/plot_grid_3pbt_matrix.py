import json
import os
import platform
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

platform_system = platform.system()
if platform_system != 'Windows':
    drive_path = r'/Users/erickmartinez/Library/CloudStorage/OneDrive-Personal'
else:
    drive_path = r'C:\Users\erick\OneDrive'

base_folder = 'Documents/ucsd/Postdoc/research/manuscripts/paper2/figure_prep/bending_tests/2023'
gc_bending_xls = 'Documents/ucsd/Postdoc/research/manuscripts/paper2/figure_prep/bending_tests/glassy_carbon_3pbt_db.xlsx'


def load_plot_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)


def normalize_path(the_path):
    global platform_system, drive_path
    if platform_system != 'Windows':
        the_path = the_path.replace('\\', '/')
    else:
        the_path = the_path.replace('/', '\\')
    return os.path.join(drive_path, the_path)


def get_file_list(base_path, extension='.csv'):
    files = []
    for f in os.listdir(base_path):
        if f.startswith('3PBT') and f.endswith('.csv'):
            files.append(f)
    return files


def main():
    global base_folder, gc_bending_xls
    base_folder = normalize_path(base_folder)
    gc_bending_xls = normalize_path(gc_bending_xls)

    gc_df = pd.read_excel(gc_bending_xls, sheet_name=1)
    gc_cols = gc_df.columns
    gc_df[gc_cols[1::]] = gc_df[gc_cols[1::]].apply(pd.to_numeric)

    gc_df = gc_df.sort_values(by=['Matrix wt %'])

    matrix_pct = gc_df['Matrix wt %'].unique()
    pane_idx = {int(mp): (int(not i < 2),  i % 2) for i, mp in enumerate(matrix_pct)}

    load_plot_style()

    fig, axes = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
    fig.set_size_inches(7., 6.0)


    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
    markers = ['o', 's', '^', 'v', 'D', '<', '>']

    mcount = {}
    mtitle_set = {}
    for mwp in matrix_pct:
        mcount[mwp] = 0
        mtitle_set[mwp] = False


    # print(gc_df[['Test ID','Matrix wt %']])

    for i, row in gc_df.iterrows():
        sid, tid, mc = row['Sample ID'], row['Test ID'], int(row['Matrix wt %'])
        pp = fr'(.*?)3PBT_{sid}\s\-\s{tid:03d}(.*?)'

        p = re.compile(pp)
        rel_path = os.path.join(base_folder, sid, 'processed_data')
        file_list = get_file_list(base_path=rel_path, extension='.csv')

        li = mcount[mc]
        for fn in file_list:
            m = p.match(fn)
            if m is not None:
                # print(m.group(0))
                bending_df: pd.DataFrame = pd.read_csv(os.path.join(rel_path, fn), comment='#')
                bending_df = bending_df.apply(pd.to_numeric)
                deformation = bending_df['Deformation (mm)'].values
                force_4cm = bending_df['Force 4cm (N)'].values
                [axi, axj] = pane_idx[mc]
                lbl1 = f'Test ID: {tid:03d}'
                axes[axi, axj].errorbar(
                    deformation, force_4cm, #yerr=0.5, xerr=0.05,
                    marker=markers[li], mfc='none', capsize=2.5,
                    ms=9, mew=1.,
                    elinewidth=1.0, lw=1.75, c=colors[li],
                    label=lbl1
                )
                if not mtitle_set[mc]:
                    axes[axi, axj].set_title(f'{mc:>2d} matrix wt %')
                mcount[mc] += 1

    for axi in axes:
        for ax in axi:
            ax.set_xlabel('Deformation (mm)')
            ax.set_ylabel('Load (N)')
            ax.legend(loc='upper left', frameon=True, fontsize=9)

    plt.show()


if __name__ == '__main__':
    main()
