import pandas as pd
from scipy.stats.distributions import t
from tkinter.filedialog import askdirectory
import numpy as np
import os
import re

pattern_clean = re.compile(r'((.*)?_CLEAN\.txt)')
pattern_coated = re.compile(r'((.*)?_COATED\.txt)')
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

    out_df = pd.DataFrame(columns=['FILE', 'SIGNAL', 'N', 'R_MEAN (V)', 'R_STD (V)', 'T.INV', 'R_SE (V)'])
    print(os.path.basename(folder))
    print('       FILE                   N   R_MEAN      R_STD    T.INV       R_SE')
    for fn in file_list:
        m_clean = pattern_clean.match(fn)
        m_coated = pattern_coated.match(fn)
        m_bgnd = pattern_bgd.match(fn)
        m_air = pattern_air.match(fn)
        if not m_clean is None:
            signal_type = 'CLEAN'

        if not m_coated is None:
            signal_type = 'COATED'

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
            'R_SE (V)': [r_se]

        }
        file = os.path.splitext(fn)[0]
        row = pd.DataFrame(data=data)
        out_df = pd.concat([out_df, row]).reset_index(drop=True)
        print(f'{file[-15:]:>10s} {signal_type:>7s} {n:>8d} {r_mean:>8.6f} {r_std:>6.4E} {tval:>8.6f} {r_se:>6.4E}')


    file_tag = os.path.basename(folder) + '_averages.csv'
    agg_df = out_df.groupby('SIGNAL').agg({'R_MEAN (V)': ['mean', 'std', standard_error, 'count'], 'R_SE (V)': [mean_err]})
    # r_mean = agg_df['R_MEAN (V)']['mean'].values
    r_se = agg_df['R_MEAN (V)']['standard_error'].values
    # r_n = agg_df['R_MEAN']['count'].values
    r_mean_error = agg_df['R_SE (V)']['mean_err'].values
    total_error = np.empty(len(agg_df))
    for i in range(len(agg_df)):
        total_error[i] = np.linalg.norm([r_se[i], r_mean_error[i]]) if not np.isinf(r_se[i]) else r_mean_error[i]
    # print(out_df)
    agg_df['Total error (V)'] = total_error
    print(agg_df)
    out_df.to_csv(os.path.join(folder, file_tag), index=False)
    agg_df.to_csv(os.path.join(folder, os.path.basename(folder) + '_aggregated.csv'), index=False)


if __name__ == '__main__':
    main()
