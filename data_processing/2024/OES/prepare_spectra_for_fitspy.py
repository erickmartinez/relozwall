import pandas as pd
import os

relative_path = r'./data/brightness_data'
outout_folder = r'./data/brightness_data_fitspy'
echelle_excel = r'./data/echelle_db.xlsx'

def main():
    global relative_path, outout_folder, echelle_excel
    # load the echelle db
    echelle_df:pd.DataFrame = pd.read_excel(echelle_excel, sheet_name=0)
    echelle_df = echelle_df[echelle_df['Label'] != 'Labsphere']
    echelle_df = echelle_df[echelle_df['Is dark'] == 0]
    echelle_df = echelle_df[echelle_df['Number of Accumulations']>=20]
    echelle_df = echelle_df.reset_index(drop=True)
    echelle_df = echelle_df[['Folder', 'File']]

    for i, row in echelle_df.iterrows():
        folder = row['Folder']
        file = row['File'].replace('.asc','.csv')
        df = pd.read_csv(os.path.join(relative_path, folder, file), comment='#')
        out_path = os.path.join(outout_folder, folder)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        df.to_csv(os.path.join(out_path, file), index=False)


if __name__ == '__main__':
    main()