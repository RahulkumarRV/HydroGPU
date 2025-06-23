import ee
import time
import os
from tqdm import tqdm
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import rasterio
import numpy as np
from rasterio.merge import merge
import re



class MergeTiffs:
    def __init__(self, input_folder, output_folder):
        """Initialize the merger with input and output directories."""
        self.input_folder = input_folder
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
    
    def find_tif_files(self):
        """Find and group TIF files by index."""
        file_groups = {}
        for file in os.listdir(self.input_folder):
            match = re.match(r"3H_Rain_(\d+)-\d{10}-\d{10}\.tif$", file)
            if match:
                index = match.group(1)
                file_groups.setdefault(index, []).append(os.path.join(self.input_folder, file))
        return file_groups
    
    def merge_and_save(self):
        """Merge grouped images and save the output."""
        file_groups = self.find_tif_files()
        
        for index, file_list in file_groups.items():
            if len(file_list) > 1:
                datasets = [rasterio.open(f) for f in file_list]
                merged_array, transform = merge(datasets)
                merged_array = merged_array.astype(np.float32)
                
                out_meta = datasets[0].meta.copy()
                out_meta.update({
                    "height": merged_array.shape[1],
                    "width": merged_array.shape[2],
                    "transform": transform,
                    "dtype": "float32"
                })
                
                output_file = os.path.join(self.output_folder, f"GEE_3H_Download_{index}_merged.tif")
                with rasterio.open(output_file, "w", **out_meta) as dest:
                    dest.write(merged_array)
                
                for ds in datasets:
                    ds.close()
                
                print(f"Merged {len(file_list)} images for index {index} -> {output_file}")
            else:
                print(f"Skipping index {index} (only 1 file found)")
        
        print("Merging completed! All images are saved in float32 format.")

class GoogleDriveDownloader:
    def __init__(self):
        """Initialize and authenticate Google Drive access."""
        self.drive = self.authenticate_drive()
    
    def authenticate_drive(self):
        """Authenticate Google Drive access using PyDrive."""
        gauth = GoogleAuth()
        gauth.LocalWebserverAuth()
        return GoogleDrive(gauth)
    
    def download_from_drive(self, folder_name, local_download_path):
        """
        Downloads all files from a specific Google Drive folder to a local directory.
        
        Parameters:
        - folder_name: The name of the folder in Google Drive where GEE data is stored.
        - local_download_path: The local directory where files will be saved.
        """
        os.makedirs(local_download_path, exist_ok=True)  # Ensure local directory exists
        
        # Get folder ID by searching for its name
        folder_query = f"title='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
        folder_list = self.drive.ListFile({'q': folder_query}).GetList()
        
        if not folder_list:
            print(f"Folder '{folder_name}' not found in Google Drive.")
            return {"status": "failure", "message": "Folder not found"}
        
        folder_id = folder_list[0]['id']  # Use the first matching folder
        
        # Get all files inside the folder
        file_list = self.drive.ListFile({'q': f"'{folder_id}' in parents"}).GetList()
        
        if not file_list:
            print(f"No files found in '{folder_name}' folder.")
            return {"status": "failure", "message": "No files in folder"}
        
        print(f"Downloading {len(file_list)} files from '{folder_name}'...")
        
        for file in file_list:
            file_name = file['title']
            file_path = os.path.join(local_download_path, file_name)
            
            print(f"Downloading {file_name}...")
            file.GetContentFile(file_path)  # Download the file
            print(f"Saved to {file_path}")
        
        print("All files downloaded successfully.")
        return {"status": "success", "message": "All files downloaded"}

class GEEDataDownloader:
    def __init__(self, project_id):
        """Initialize Google Earth Engine authentication and set project ID."""
        self.project_id = project_id
        self.authenticate_gee()
        self.gdriveDownlaoder = GoogleDriveDownloader()
        self.temp_folder = 'downloaded_GEE_rain'
    
    def authenticate_gee(self):
        """Authenticate and initialize Google Earth Engine."""
        ee.Authenticate()
        ee.Initialize(project=self.project_id)
    
    def sum_every_3_images(self, collection):
        """Function to sum every 3 consecutive images."""
        size = collection.size()
        num_groups = size.divide(3).floor()
        
        def sum_group(i):
            start = ee.Number(i).multiply(3)
            images = collection.toList(size).slice(start, start.add(3))
            return ee.ImageCollection(images).sum().set('group', i)
        
        summed_collection = ee.ImageCollection(ee.List.sequence(0, num_groups.subtract(1)).map(sum_group))
        return summed_collection
    
    def execute_gee_tasks(self, dataset_id, asset_id, start_date, end_date, drive_folder, local_download_path):
        """
        Download dataset from Google Earth Engine for a given geometry and time range.
        """

        try:
            geometry = ee.FeatureCollection(asset_id)
            
            collection = ee.ImageCollection(dataset_id)\
                .filterBounds(geometry)\
                .filterDate(ee.Date(start_date), ee.Date(end_date))\
                .select('hourlyPrecipRate')
            
            print(f"Total Images in Collection: {collection.size().getInfo()}")
            
            collection_3h = self.sum_every_3_images(collection)
            
            def export_image(image, index):
                task = ee.batch.Export.image.toDrive(
                    image=image.clip(geometry),
                    description=f"GEE_3H_Download_{index}",
                    folder=drive_folder,
                    scale=30,
                    region=geometry.geometry().bounds(),
                    maxPixels=1e13
                )
                task.start()
                return task
            
            tasks = []
            for i in range(collection_3h.size().getInfo()):
                image = ee.Image(collection_3h.toList(collection_3h.size()).get(i))
                tasks.append(export_image(image, i))
            
            print("Exports started. Check Google Drive for completion.")
            

            with tqdm(total=len(tasks), desc="GEE tasks", unit="task") as pbar:
                for task in tasks:
                    while task.active():
                        time.sleep(20)
                    pbar.update(1)
            
            print("All exports completed.")

            self.gdriveDownlaoder.download_from_drive(drive_folder, self.temp_folder)
            
            mergeTiffs = MergeTiffs(self.temp_folder, local_download_path)
            mergeTiffs.merge_and_save()
            
        except Exception as e:
            print(f"Error occurred: {e}") 


                                                                                                      


# Example usage
# if __name__ == "__main__":
#     project_id = 'ee-rahul-kumar'
#     dataset_id = 'JAXA/GPM_L3/GSMaP/v6/operational'
#     asset_id = 'projects/ee-rahul-kumar/assets/Lower_ganga'
#     start_date = '2023-07-01'
#     end_date = '2023-07-02'
#     drive_folder = 'LGB_2023_07_01_rain'
    
#     downloader = GEEDataDownloader(project_id)
#     downloader.execute_gee_tasks(dataset_id, asset_id, start_date, end_date, drive_folder)
