import xml.etree.ElementTree as ET

def main():
    xml_doc = ET.parse('colorMaps.xml')
    # get the root element
    root = xml_doc.getroot()
    doc = ET.Element('ColorMaps')
    for element in root.iter('ColorMap'):
        name = element.attrib["name"].lower()
        space = element.attrib["space"].lower()
        cmap = f"{name.replace(' ', '_')}"
        new_cmap = ET.SubElement(doc, "ColorMap")
        new_cmap.attrib["Name"] = cmap
        new_cmap.attrib["space"] = space
        points = element.iter("Point")
        for i, pt in enumerate(points):
            entry = ET.SubElement(new_cmap, 'ColorMapEntry')
            r = float(pt.attrib['r'])
            g = float(pt.attrib['g'])
            b = float(pt.attrib['b'])
            o = float(pt.attrib['o'])
            # entry.attrib['RGB'] = f"{r:.3f}, {g:.3f}, {b:.3f}"
            entry.attrib['r'] = f"{r:.3f}"
            entry.attrib['g'] = f"{g:.3f}"
            entry.attrib['b'] = f"{b:.3f}"
            entry.attrib['o'] = f"{o:.3f}"
            entry.attrib['Index'] = str(i)
    tree = ET.ElementTree(doc)
    tree.write("paraview_color_maps.xml")



if __name__ == '__main__':
    main()