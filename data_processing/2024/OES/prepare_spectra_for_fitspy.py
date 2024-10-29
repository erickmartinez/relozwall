import pandas as pd
import os

relative_path = r'./data/brightness_data'
outout_folder = r'./data/brightness_data_fitspy'

def main():
    global relative_path, outout_folder
    folders = os.listdir(relative_path)
    for folder in folders:
        files = [f for f in os.listdir(os.path.join(relative_path, folder)) if f.endswith('.csv')]
        for file in files:
            df = pd.read_csv(os.path.join(relative_path, folder, file), comment='#')
            out_path = os.path.join(outout_folder, folder)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            df.to_csv(os.path.join(out_path, file), index=False)


if __name__ == '__main__':
    main()