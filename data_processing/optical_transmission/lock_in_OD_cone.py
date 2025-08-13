import pandas as pd
from scipy.stats.distributions import t
from tkinter.filedialog import askdirectory
import numpy as np
import os
import re

pattern_clean = re.compile(r'((.*)?_CLEAN\.txt)')
pattern_coated = re.compile(r'((.*)?_X(\d+)_(\d+)MM_COATED\.txt)')
pattern_air = re.compile(r'((.*)?_AIR\.txt)')
pattern_bgd = re.compile(r'((.*)?_BGND\.txt)')

def mean_err(x):
    x = np.array(x)
    n = len(x)
    return np.linalg.norm(x) / n

def standard_error(x):
    n = len(x)
    if n == 1:
        return np.inf
    std = np.std(x, ddof=1)
    confidence = 0.95
    alpha = 1 - confidence
    tval = t.ppf(1 - alpha/2, n-1)
    return std * tval / np.sqrt(n)

def main():
    folder = askdirectory()
    file_list = [fn for fn in os.listdir(folder) if fn.endswith('.txt')]
    confidence = 0.95
    alpha = 1. - confidence

    out_df = pd.DataFrame(columns=['FILE', 'SIGNAL', 'N', 'R_MEAN (V)', 'R_STD (V)', 'T.INV', 'R_SE (V)', 'X (mm)'])
    print(os.path.basename(folder))
    print('       FILE                   N   R_MEAN      R_STD    T.INV       R_SE  X (MM)')

    for fn in file_list:
        m_clean = pattern_clean.match(fn)
        m_coated = pattern_coated.match(fn)
        m_bgnd = pattern_bgd.match(fn)
        m_air = pattern_air.match(fn)
        x = 0.0

        if not m_clean is None:
            signal_type = 'CLEAN'

        if m_coated:
            signal_type = 'COATED'
            x = int(m_coated.group(4))

        if not m_bgnd is None:
            signal_type = 'BGND'

        if m_air:
            signal_type = 'AIR'

        df = pd.read_csv(os.path.join(folder, fn), delimiter='\t').apply(pd.to_numeric)
        r_val = df['R[V]'].values
        n = len(r_val)
        tval = t.ppf(1. - alpha / 2., n - 1)
        r_mean = r_val.mean()
        r_std = np.std(r_val, ddof=1)
        r_se = r_std * tval / np.sqrt(n)
        data = {
            'FILE': [fn],
            'SIGNAL': [signal_type],
            'N': [n],
            'R_MEAN (V)': [r_mean],
            'R_STD (V)': [r_std],
            'T.INV': [tval],
            'R_SE (V)': [r_se],
            'X (mm)': [x]
        }
        file = os.path.splitext(fn)[0]
        row = pd.DataFrame(data=data)
        out_df = pd.concat([out_df, row]).reset_index(drop=True)
        print(f'{file[-15:]:>10s} {signal_type:>7s} {n:>8d} {r_mean:>8.6f} {r_std:>6.4E} {tval:>8.6f} {r_se:>6.4E} {x:>6.0f}')


    file_tag = os.path.basename(folder) + '_averages.csv'
    clean_df = out_df[out_df['SIGNAL'] == 'CLEAN']
    r_mean_clean = clean_df['R_MEAN (V)'].values
    r_n_clean = clean_df['N'].values
    r_se_clean = clean_df['R_SE (V)'].values
    r_mean_clean_repetitions = r_mean_clean.mean()
    r_se_clean_repetitions = standard_error(r_mean_clean)
    r_se_mean = mean_err(r_se_clean)
    total_error = np.linalg.norm([r_se_clean_repetitions, r_se_mean])

    aggregated_df =   out_df[~(out_df['SIGNAL'] == 'CLEAN')]
    clean_row = pd.DataFrame(data={
        'FILE': ['AGGREGATED'],
        'SIGNAL': ['CLEAN'],
        'N': [r_n_clean.sum()],
        'R_MEAN (V)': [r_mean_clean_repetitions],
        'R_STD (V)': [r_mean_clean.std(ddof=1)],
        'T.INV': [t.ppf(1. - alpha / 2., r_n_clean.sum() - 1)],
        'R_SE (V)': [total_error],
        'X (mm)': [out_df['X (mm)'].max()]
    })
    aggregated_df = pd.concat([aggregated_df, clean_row],ignore_index=True).reset_index(drop=True)


    print(aggregated_df)
    out_df.to_csv(os.path.join(folder, file_tag), index=False)
    aggregated_df.to_csv(os.path.join(folder, os.path.basename(folder) + '_aggregated.csv'), index=False)


if __name__ == '__main__':
    main()
