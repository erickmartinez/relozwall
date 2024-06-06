import pandas as pd
from scipy.stats.distributions import t
from tkinter.filedialog import askdirectory
import numpy as np
import os
import re

pattern_clean = re.compile(r'((.*)?_CLEAN\.txt)')
pattern_coated = re.compile(r'((.*)?_COATED\.txt)')
pattern_bgd = re.compile(r'((.*)?_BGND\.txt)')

def main():
    folder = askdirectory()
    file_list = [fn for fn in os.listdir(folder) if fn.endswith('.txt')]
    confidence = 0.95
    alpha = 1. - confidence

    out_df = pd.DataFrame(columns=['CLEAN/COATED', 'N', 'R_MEAN (V)', 'R_STD (V)', 'T.INV', 'R_SE (V)'])
    print(os.path.basename(folder))
    print('               N   R_MEAN      R_STD    T.INV       R_SE')
    for fn in file_list:
        m_clean = pattern_clean.match(fn)
        m_coated = pattern_coated.match(fn)
        m_bgnd = pattern_bgd.match(fn)
        if not m_clean is None:
            clean_or_coated = 'CLEAN'

        if not m_coated is None:
            clean_or_coated = 'COATED'

        if not m_bgnd is None:
            clean_or_coated = 'BGND'

        df = pd.read_csv(os.path.join(folder, fn), delimiter='\t').apply(pd.to_numeric)
        r_val = df['R[V]'].values
        n = len(r_val)
        tval = t.ppf(1. - alpha / 2., n - 1)
        r_mean = r_val.mean()
        r_std = np.std(r_val, ddof=1)
        r_se = r_std * tval / np.sqrt(n)
        data = {
            'CLEAN/COATED': [clean_or_coated],
            'N': [n],
            'R_MEAN (V)': [r_mean],
            'R_STD (V)': [r_std],
            'T.INV': [tval],
            'R_SE (V)': [r_se]

        }
        row = pd.DataFrame(data=data)
        out_df = pd.concat([out_df, row]).reset_index(drop=True)
        print(f'{clean_or_coated:>7s} {n:>8d} {r_mean:>8.6f} {r_std:>6.4E} {tval:>8.6f} {r_se:>6.4E}')

    file_tag = os.path.basename(folder) + '_averages.csv'
    # print(out_df)
    out_df.to_csv(os.path.join(folder, file_tag), index=False)


if __name__ == '__main__':
    main()
