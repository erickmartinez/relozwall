import pandas as pd
import xml.etree.ElementTree as ET
import os

csv = 'rainbow_uniform.csv'

def main():
    global csv
    df: pd.DataFrame = pd.read_csv(csv, usecols=['R', 'G', 'B']).apply(pd.to_numeric)
    xml_doc = ET.parse('paraview_color_maps.xml')
    # get the root element
    root = xml_doc.getroot()
    doc = ET.Element('ColorMaps')
    cmap = ET.SubElement(root, 'ColorMap')
    tag = os.path.splitext(csv)[0]
    cmap.attrib['Name'] = tag.lower()
    cmap.attrib['space'] = 'rgb'
    for i, row in df.iterrows():
        entry = ET.SubElement(cmap, 'ColorMapEntry')
        entry.attrib['r'] = f"{row['R']:.3f}"
        entry.attrib['g'] = f"{row['G']:.3f}"
        entry.attrib['b'] = f"{row['B']:.3f}"
    # tree = ET.ElementTree(doc)
    xml_doc.write("paraview_color_maps.xml")


if __name__ == '__main__':
    main()