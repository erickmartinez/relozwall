import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import json
from matplotlib.ticker import ScalarFormatter


if __name__ == "__main__":
    calibration_df = pd.read_csv('https://raw.githubusercontent.com/erickmartinez/relozwall/main/ir_thermography/optronics_OL_455-12-2_SN_96203007_calibration_table.csv')