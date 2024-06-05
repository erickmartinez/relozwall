import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
import xml.etree.ElementTree as ET
# import vtkplotlib as vpl
import numpy as np
import os
import json

# mycmap = cm.make_cmap('_colorMaps.py') #make the Matplotlib compatible colormap ## to use colormap:

color_map = 'rainbow_uniform'
t_range = [0., 1.]
label = 'Damage'

def load_plot_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)

def main():
    global color_map, t_range
    tree = ET.parse('./colormaps/paraview_color_maps.xml')
    root = tree.getroot()
    doc = ET.Element('ColorMaps')
    paraview_color_maps = [cm.attrib["Name"] for cm in root.iter('ColorMap')]
    # for pcm in paraview_color_maps:
    #     print(pcm)
    if color_map in paraview_color_maps:
        color_list = []
        color_map_entry = root.findall(f'./ColorMap[@Name=\'{color_map}\']')[0]
        for element in color_map_entry.iter("ColorMapEntry"):
            r = float(element.attrib['r'])
            g = float(element.attrib['g'])
            b = float(element.attrib['b'])
            # o = float(element.attrib['o'])
            rgb = (r, g, b)
            color_list.append(rgb)
        cmap = colors.LinearSegmentedColormap.from_list(color_map,
                                                        color_list,
                                                        N=256)
    else:
        cmap = plt.colormaps.get_cmap(color_map)

    # cmap = vpl.colors.cm.get_cmap(color_map)
    load_plot_style()
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4., 1.)

    norm1 = plt.Normalize(vmin=t_range[0], vmax=t_range[1])
    cb = mpl.colorbar.ColorbarBase(
        ax, orientation='horizontal',
        cmap=cmap,
        norm=norm1,
        # extend='min'
    )
    ax.tick_params(labelsize=14)
    ax.set_xlabel(label, fontsize=16)
    fig.savefig(f'{color_map}_{t_range[0]:.1f}-{t_range[1]:.1f}.svg', dpi=600)

    plt.show()

if __name__ == '__main__':
    main()
