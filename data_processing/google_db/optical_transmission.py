import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pandas as pd
import platform

drive_path = r'C:\Users\erick\OneDrive' if platform.system() == 'Windows' else (r'/Users/erickmartinez/Library'
                                                                                r'/CloudStorage/OneDrive-Personal')

base_path = r'Documents\ucsd\Postdoc\research\data\firing_tests'
database_filetag = 'optical_transmission_db'

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']

# The ID and range of a sample spreadsheet.
TRANSMISSION_SAMPLE_SPREADSHEET_ID = '1uFb3XkRsN0h-rs6B42Sv3rrbc5rITlInAKrejnQTrJI'
LASER_SAMPLE_RANGE_NAME = 'thicknesses!A:V'

RECIPE_SAMPLE_SPREADSHEET_ID = '1MOLyrv2BX5GRrboMMgVXZA9SU4779zFoSvm4sHrJt4k'
RECIPE_SAMPLE_RANGE_NAME = 'Recipe 4!A:W'


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
