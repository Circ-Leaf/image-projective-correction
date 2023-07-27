from googleAPI import GoogleAPI

googleAPI = GoogleAPI(drive_token_path='./auth_keys/drive/token.json', docs_token_path='./auth_keys/docs/token.json',
                drive_credentials_path='./auth_keys/drive/credentials.json', docs_credentials_path='./auth_keys/docs/credentials.json')
print(googleAPI.OCR("picture/picture_1_cut.png"))