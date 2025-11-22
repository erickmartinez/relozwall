import json
from matplotlib import rcParams
import os


def load_plot_style(font: str = None):
    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to plot_style.json
    style_path = os.path.join(current_dir, 'plot_style.json')
    with open(style_path, 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    rcParams.update(plot_style)
    if font:
        rcParams['font.family'] = font
    rcParams['text.latex.preamble'] = (r'\usepackage{mathptmx}'
                                           r'\usepackage{xcolor}'
                                           r'\usepackage{helvet}'
                                           r'\usepackage{siunitx}'
                                           r'\usepackage{amsmath, array, makecell}')