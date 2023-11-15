import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pandas as pd

# base_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests'
base_path = '/Users/erickmartinez/Documents/UCSD/research/data/firing tests'
database_filetag = 'merged_db'

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']

# The ID and range of a sample spreadsheet.
LASER_SAMPLE_SPREADSHEET_ID = '1Ga9_mzYaTId76LA3Kp2ZczzjOPLkiJcfy_B3jgwjznM'
LASER_SAMPLE_RANGE_NAME = 'Tests!A:AA'

RECIPE_SAMPLE_SPREADSHEET_ID = '1MOLyrv2BX5GRrboMMgVXZA9SU4779zFoSvm4sHrJt4k'
RECIPE_SAMPLE_RANGE_NAME = 'Recipe 4!A:W'

MERGED_COLUMNS = [
    'ROW', 'Sample code', 'Power percent setting (%)', 'Irradiation time (s)', 'Pressure (mTorr)', 'Density (g/cm^3)',
    'Density error (g/cm^3)', 'Sample diameter (cm)', 'Diameter error (cm)', 'Recession rate (cm/s)',
    'Recession rate error (cm/s)', 'Quality', 'Spheres (wt %)', 'Filler (wt %)', 'Binder (wt %)', 'Resin (wt %)',
    'Hardener (wt %)', 'Type of spheres', 'Type of filler', 'Type of hardener', 'Baking temperature (°C)',
    'Baking time (min)', 'Ramping rate (°C/min)', 'Cast tube length (cm)', 'Cast tube ID (cm)'
]

NUMERIC_COLS = [
    'ROW', 'Power percent setting (%)', 'Irradiation time (s)', 'Pressure (mTorr)', 'Density (g/cm^3)',
    'Density error (g/cm^3)', 'Sample diameter (cm)', 'Diameter error (cm)', 'Recession rate (cm/s)',
    'Recession rate error (cm/s)', 'Spheres (wt %)', 'Filler (wt %)', 'Binder (wt %)', 'Resin (wt %)',
    'Hardener (wt %)', 'Baking temperature (°C)', 'Baking time (min)', 'Cast tube length (cm)', 'Quality',
    'Ramping rate (°C/min)', 'Cast tube ID (cm)'
]

# DISCARD_ROWS = [
#     173, 174
# ]

def pull_sheet_data(spreadsheet_id, sheet_range):
    """Shows basic usage of the Sheets API.
        Prints values from a sample spreadsheet.
        """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    try:
        service = build('sheets', 'v4', credentials=creds)

        # Call the Sheets API
        sheet = service.spreadsheets()
        result = sheet.values().get(spreadsheetId=spreadsheet_id,
                                    range=sheet_range).execute()
        values = result.get('values', [])

        if not values:
            print('No data found.')
            return
        else:
            # rows = sheet.values().get(spreadsheetId=spreadsheet_id,
            #                           range=sheet_range).execute()
            # data = rows.get('values')
            # print("COMPLETE: Data copied")
            # return data
            # for i in range(len(values)):
            #     print(f"len row {i}: {len(values[i])}")

            df = pd.DataFrame(values[1:], columns=values[0], )
            return df

    except HttpError as err:
        print(err)
        return


if __name__ == '__main__':
    df_lasers = pull_sheet_data(LASER_SAMPLE_SPREADSHEET_ID, LASER_SAMPLE_RANGE_NAME)
    df_lasers['ROW'] = df_lasers.index + 2
    df_recipe =  pull_sheet_data(RECIPE_SAMPLE_SPREADSHEET_ID, RECIPE_SAMPLE_RANGE_NAME)
    df_merged = pd.merge(df_lasers, df_recipe, on=["Sample code"], how="left", validate="many_to_many")
    df_merged = df_merged[MERGED_COLUMNS]
    df_merged[NUMERIC_COLS] = df_merged[NUMERIC_COLS].apply(pd.to_numeric)
    df_merged = df_merged[df_merged['Spheres (wt %)'].notna()]
    df_merged = df_merged[df_merged['Quality'] > 0 ]
    # df_merged.to_csv(
    #     r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\merged_db.csv', index=False,
    #     encoding="utf-8"
    # )
    df_merged.to_excel(
        os.path.join(base_path, database_filetag + '.xlsx'),
        sheet_name='Laser tests',
        index=False
    )

    # df = pd.DataFrame(gdata[1:], columns=gdata[1,:])
    print(df_merged)

