from tkinter.filedialog import askdirectory
import pandas as pd
import pypdf
import os
import re
from datetime import datetime

date_pattern =  re.compile(r'(\d{2}\/\d{2}\/\d{4})')
item_pattern = re.compile(r'\n(\d)(\d+)\s(.*?)\$(\d+\.\d+)\s(\d+\.\d+)\s\$(\d+\.\d+)')

def main():
    folder = askdirectory()
    now = datetime.now()
    file_list = [f for f in os.listdir(folder) if f.endswith('.pdf')]
    df = pd.DataFrame(columns=[
        'Item', 'SKU', 'Vendor', 'Date added', 'Date updated', 'Allocation', 'Stock',
        'Assigned to', 'Price', 'Purchase date', 'Notes'
    ])
    for fn in file_list:
        full_file = os.path.join(folder, fn)
        print(full_file)
        with open(full_file, 'rb') as pdf_file:
            reader = pypdf.PdfReader(pdf_file)
            pages = reader.pages
            text = ''
            date_str = ''
            for i, p in enumerate(pages):
                text += p.extract_text()
                if i == 0:
                    match_date = date_pattern.findall(text)
                    date_str = match_date[0]
                match_items = item_pattern.findall(text)
                now_str = now.strftime('%m/%d/%Y')
                for m in match_items:
                    df = pd.concat([
                        df,
                        pd.DataFrame(data={
                            'Item': [m[2]],
                            'SKU': [m[1]],
                            'Vendor': ['Department of Chemistry & Biochemistry Storehouse'],
                            'Date added': [now_str],
                            'Date updated': [now_str],
                            'Allocation': ['Supplies, Task 1'],
                            'Stock': [int(m[4][0])],
                            'Assigned to': ['Erick'],
                            'Price': [m[5]],
                            'Purchase date': [date_str],
                            'Notes': ['Purchased with Oracle Financials, Unit price ($): ' + m[3] + f' #: {int(m[4][0])}'],
                        })
                    ])

    df.sort_values(by=['Purchase date'], ascending=True, inplace=True)
    print(df)
    output_fn = 'new_inventory_' + now.strftime('%Y-%m-%d') + '.csv'
    df.to_csv(os.path.join(folder, output_fn), index=False)



if __name__ == '__main__':
    main()