from __future__ import print_function
import pickle
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from apiclient.http import MediaFileUpload
from apiclient.http import MediaIoBaseUpload
from apiclient import errors 
import io
import numpy as np
from os import path
# If modifying these scopes, delete the file token.pickle.

class Session:

    def generate_random_seed(self):
        page_token = None
        while True:
            response = self.service.files().list(q="name = 'seeds.txt'",
                                                  spaces='drive',
                                                  fields='nextPageToken, files(id, name)',
                                                  pageToken=page_token).execute()
            for file in response.get('files', []):
                print('Found file: %s (%s)' % (file.get('name'), file.get('id')))
                seed_file = file.get('id')

            page_token = response.get('nextPageToken', None)
            if page_token is None:
                break

        text = self.service.files().get_media(fileId=seed_file).execute().decode('UTF-8')
        seeds_used = [int(seed) for seed in text.split('\n')]
        new_seed = seeds_used[-1] + 1

        np.save("current_seed.npy", new_seed)
        text += ('\n%d' %(new_seed))

        file_metadata = {'name': 'seeds.txt'}
        media = MediaIoBaseUpload(io.BytesIO(text.encode('utf-8')), mimetype="text/plain")
        file = self.service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        try:
            self.service.files().delete(fileId=seed_file).execute() # delete old seeds.txt
        except errors.HttpError:
            print("File already deleted/overwritten.")

        return new_seed

    def get_folder(self):
        page_token = None
        while True:
            response = self.service.files().list(q= "mimeType = 'application/vnd.google-apps.folder' and name = 'Map %d'" % (self.map_num),
                                                  spaces='drive',
                                                  fields='nextPageToken, files(id, name)',
                                                  pageToken=page_token).execute()
            for file in response.get('files', []):
                print('Found file: %s (%s)' % (file.get('name'), file.get('id')))
                return file.get('id')

            page_token = response.get('nextPageToken', None)
            if page_token is None:
                break

    def __init__(self, map_num, imt = False):
        self.map_num = map_num
        self.SCOPES = ['https://www.googleapis.com/auth/drive']
        self.imt = imt
        creds = None
        # The file token.pickle stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)

        self.service = build('drive', 'v3', credentials=creds)

        self.folder_id = self.get_folder()

        # look for random seed already existing
        if (path.exists("current_seed.npy")):
            self.random_seed = np.load("current_seed.npy")
        else:
            self.random_seed = self.generate_random_seed()

    def reset_seed(self):
        np.random.seed(2 * self.random_seed)


    def save_data(self):
        if (self.imt):
            file_metadata = {'name': 'data%d_imt.npy' % (self.random_seed), 'parents': [self.folder_id]}
            media = MediaFileUpload('data/data%d_imt.npy' % (self.random_seed), mimetype=None)
        else:
            file_metadata = {'name': 'data%d_ital.npy' % (self.random_seed), 'parents': [self.folder_id]}
            media = MediaFileUpload('data/data%d_ital.npy' % (self.random_seed), mimetype=None)

        file = self.service.files().create(body=file_metadata,
                                                media_body=media,
                                                fields='id').execute()
        print('Successfully saved! File ID: %s' % file.get('id'))

if __name__ == '__main__':
    s = Session(3)
