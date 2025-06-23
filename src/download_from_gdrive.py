import time
import os
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


def authenticate_drive():
    """Authenticate Google Drive access using PyDrive."""
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    return GoogleDrive(gauth)



def download_from_drive(drive, folder_name, local_download_path):
    """
    Downloads all files from a specific Google Drive folder to a local directory.

    Parameters:
    - drive: Authenticated GoogleDrive instance
    - folder_name: The name of the folder in Google Drive where GEE data is stored
    - local_download_path: The local directory where files will be saved
    """
    # Ensure local directory exists
    os.makedirs(local_download_path, exist_ok=True)

    # Get folder ID by searching for its name
    folder_query = f"title='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
    folder_list = drive.ListFile({'q': folder_query}).GetList()
    
    if not folder_list:
        print(f"Folder '{folder_name}' not found in Google Drive.")
        return

    folder_id = folder_list[0]['id']  # Use the first matching folder

    # Get all files inside the folder
    file_list = drive.ListFile({'q': f"'{folder_id}' in parents"}).GetList()

    if not file_list:
        print(f"No files found in '{folder_name}' folder.")
        return

    print(f"Downloading {len(file_list)} files from '{folder_name}'...")

    for file in file_list:
        file_name = file['title']
        file_path = os.path.join(local_download_path, file_name)

        print(f"Downloading {file_name}...")
        file.GetContentFile(file_path)  # Download the file
        print(f"Saved to {file_path}")

    print("All files downloaded successfully.")



drive = authenticate_drive()

drive_folder = "Lower_Ganga_3H_Rain_2023-07-01_2023-10-01"

# Local directory to save the downloaded images
local_directory = "./downloaded_GEE_rain"

download_from_drive(drive, drive_folder, local_directory)